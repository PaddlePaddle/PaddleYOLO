# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from ppdet.core.workspace import register

from ..bbox_utils import batch_distance2bbox
from ..assigners.utils import generate_anchors_for_grid_cell
from ppdet.modeling.backbones.csp_darknet import BaseConv
from ppdet.modeling.layers import MultiClassNMS

__all__ = ['YOLOv8Head', 'YOLOv8InsHead']


@register
class YOLOv8Head(nn.Layer):
    __shared__ = [
        'num_classes', 'eval_size', 'act', 'trt', 'exclude_nms',
        'exclude_post_process'
    ]
    __inject__ = ['assigner', 'nms']

    def __init__(self,
                 num_classes=80,
                 in_channels=[256, 512, 1024],
                 act='silu',
                 fpn_strides=[8, 16, 32],
                 grid_cell_scale=5.0,
                 grid_cell_offset=0.5,
                 reg_max=16,
                 reg_range=None,
                 use_varifocal_loss=False,
                 assigner='TaskAlignedAssigner',
                 nms='MultiClassNMS',
                 eval_size=[640, 640],
                 loss_weight={
                     'class': 1.0,
                     'iou': 2.5,
                     'dfl': 0.5,
                 },
                 trt=False,
                 exclude_nms=False,
                 exclude_post_process=False,
                 print_l1_loss=True):
        super(YOLOv8Head, self).__init__()
        assert len(in_channels) > 0, "len(in_channels) should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        if reg_range:
            self.reg_range = reg_range
        else:
            self.reg_range = (0, reg_max)  # not reg_max+1
        self.reg_channels = self.reg_range[1] - self.reg_range[0]
        self.use_varifocal_loss = use_varifocal_loss
        self.assigner = assigner
        self.nms = nms
        if isinstance(self.nms, MultiClassNMS) and trt:
            self.nms.trt = trt
        self.eval_size = eval_size
        self.loss_weight = loss_weight
        self.exclude_nms = exclude_nms
        self.exclude_post_process = exclude_post_process
        self.print_l1_loss = print_l1_loss

        # cls loss
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=paddle.to_tensor([1.0]), reduction="mean")

        # pred head
        c2 = max((16, in_channels[0] // 4, self.reg_channels * 4))
        c3 = max(in_channels[0], self.num_classes)
        self.conv_reg = nn.LayerList()
        self.conv_cls = nn.LayerList()
        for in_c in self.in_channels:
            self.conv_reg.append(
                nn.Sequential(* [
                    BaseConv(
                        in_c, c2, 3, 1, act=act),
                    BaseConv(
                        c2, c2, 3, 1, act=act),
                    nn.Conv2D(
                        c2,
                        self.reg_channels * 4,
                        1,
                        bias_attr=ParamAttr(regularizer=L2Decay(0.0))),
                ]))
            self.conv_cls.append(
                nn.Sequential(* [
                    BaseConv(
                        in_c, c3, 3, 1, act=act),
                    BaseConv(
                        c3, c3, 3, 1, act=act),
                    nn.Conv2D(
                        c3,
                        self.num_classes,
                        1,
                        bias_attr=ParamAttr(regularizer=L2Decay(0.0))),
                ]))
        # projection conv
        self.dfl_conv = nn.Conv2D(self.reg_channels, 1, 1, bias_attr=False)
        self.dfl_conv.skip_quant = True
        self.proj = paddle.linspace(0, self.reg_channels - 1, self.reg_channels)
        self.dfl_conv.weight.set_value(
            self.proj.reshape([1, self.reg_channels, 1, 1]))
        self.dfl_conv.weight.stop_gradient = True

        # self._init_bias()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def _init_bias(self):
        for a, b, s in zip(self.conv_reg, self.conv_cls, self.fpn_strides):
            a[-1].bias.set_value(1.0)  # box
            b[-1].bias[:self.num_classes] = math.log(5 / self.num_classes /
                                                     (640 / s)**2)

    def forward(self, feats, targets=None):
        if self.training:
            return self.forward_train(feats, targets)
        else:
            return self.forward_eval(feats)

    def forward_train(self, feats, targets):
        anchors, anchor_points, num_anchors_list, stride_tensor = \
            generate_anchors_for_grid_cell(
                feats, self.fpn_strides, self.grid_cell_scale,
                self.grid_cell_offset)

        cls_score_list, reg_distri_list = [], []
        for i, feat in enumerate(feats):
            reg_distri = self.conv_reg[i](feat)
            cls_logit = self.conv_cls[i](feat)
            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.flatten(2).transpose([0, 2, 1]))
            reg_distri_list.append(reg_distri.flatten(2).transpose([0, 2, 1]))
        cls_score_list = paddle.concat(cls_score_list, axis=1)
        reg_distri_list = paddle.concat(reg_distri_list, axis=1)

        return self.get_loss([
            cls_score_list, reg_distri_list, anchors, anchor_points,
            num_anchors_list, stride_tensor
        ], targets)

    def forward_eval(self, feats):
        anchor_points, stride_tensor = self._generate_anchors(feats)

        cls_score_list, reg_dist_list = [], []
        for i, feat in enumerate(feats):
            _, _, h, w = feat.shape
            l = h * w
            reg_dist = self.conv_reg[i](feat)
            cls_logit = self.conv_cls[i](feat)

            reg_dist = reg_dist.reshape(
                [-1, 4, self.reg_channels, l]).transpose(
                    [0, 2, 3, 1])  # Note diff
            reg_dist = self.dfl_conv(F.softmax(reg_dist, axis=1)).squeeze(1)

            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.reshape([-1, self.num_classes, l]))
            reg_dist_list.append(reg_dist)

        cls_score_list = paddle.concat(cls_score_list, axis=-1)
        reg_dist_list = paddle.concat(reg_dist_list, axis=1)

        return cls_score_list, reg_dist_list, anchor_points, stride_tensor

    def _generate_anchors(self, feats=None, dtype='float32'):
        # just use in eval time
        anchor_points = []
        stride_tensor = []
        for i, stride in enumerate(self.fpn_strides):
            if feats is not None:
                _, _, h, w = feats[i].shape
            else:
                h = int(self.eval_size[0] / stride)
                w = int(self.eval_size[1] / stride)
            shift_x = paddle.arange(end=w) + self.grid_cell_offset
            shift_y = paddle.arange(end=h) + self.grid_cell_offset
            shift_y, shift_x = paddle.meshgrid(shift_y, shift_x)
            anchor_point = paddle.cast(
                paddle.stack(
                    [shift_x, shift_y], axis=-1), dtype=dtype)
            anchor_points.append(anchor_point.reshape([-1, 2]))
            stride_tensor.append(paddle.full([h * w, 1], stride, dtype=dtype))
        anchor_points = paddle.concat(anchor_points)
        stride_tensor = paddle.concat(stride_tensor)
        return anchor_points, stride_tensor

    def get_loss(self, head_outs, targets):
        raise NotImplementedError('YOLOv8 training not implemented yet.')
        return {
            'loss': paddle.zeros([1]),
            'loss_cls': paddle.zeros([1]),
            'loss_box': paddle.zeros([1]),
        }

    def post_process(self,
                     head_outs,
                     im_shape,
                     scale_factor,
                     infer_shape=[640, 640]):
        pred_scores, pred_dist, anchor_points, stride_tensor = head_outs
        pred_bboxes = batch_distance2bbox(anchor_points, pred_dist)
        pred_bboxes *= stride_tensor

        if self.exclude_post_process:
            return paddle.concat(
                [pred_bboxes, pred_scores.transpose([0, 2, 1])], axis=-1)
        else:
            # scale bbox to origin
            scale_factor = scale_factor.flip(-1).tile([1, 2]).unsqueeze(1)
            pred_bboxes /= scale_factor
            if self.exclude_nms:
                # `exclude_nms=True` just use in benchmark
                return pred_bboxes, pred_scores
            else:
                bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
                return bbox_pred, bbox_num


class MaskProto(nn.Layer):
    # YOLOv8 mask Proto module for instance segmentation models
    def __init__(self, ch_in, num_protos=256, num_masks=32, act='silu'):
        super().__init__()
        self.conv1 = BaseConv(ch_in, num_protos, 3, 1, act=act)
        self.upsample = nn.Conv2DTranspose(
            num_protos, num_protos, 2, 2, 0, bias_attr=True)
        self.conv2 = BaseConv(num_protos, num_protos, 3, 1, act=act)
        self.conv3 = BaseConv(num_protos, num_masks, 1, 1, act=act)

    def forward(self, x):
        return self.conv3(self.conv2(self.upsample(self.conv1(x))))


@register
class YOLOv8InsHead(YOLOv8Head):
    __shared__ = [
        'num_classes', 'width_mult', 'eval_size', 'act', 'trt', 'exclude_nms',
        'exclude_post_process', 'with_mask'
    ]
    __inject__ = ['assigner', 'nms']

    def __init__(self,
                 with_mask=True,
                 num_classes=80,
                 num_masks=32,
                 num_protos=256,
                 width_mult=1.0,
                 in_channels=[256, 512, 1024],
                 act='silu',
                 fpn_strides=[8, 16, 32],
                 grid_cell_scale=5.0,
                 grid_cell_offset=0.5,
                 reg_max=16,
                 reg_range=None,
                 use_varifocal_loss=False,
                 assigner='TaskAlignedAssigner',
                 nms='MultiClassNMS',
                 eval_size=[640, 640],
                 loss_weight={
                     'class': 1.0,
                     'iou': 2.5,
                     'dfl': 0.5,
                 },
                 mask_thr_binary=0.5,
                 trt=False,
                 exclude_nms=False,
                 exclude_post_process=False,
                 print_l1_loss=True):
        super(YOLOv8InsHead, self).__init__()
        assert len(in_channels) > 0, "len(in_channels) should > 0"
        self.with_mask = with_mask
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_masks = num_masks
        self.num_protos = int(num_protos * width_mult)
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        if reg_range:
            self.reg_range = reg_range
        else:
            self.reg_range = (0, reg_max)  # not reg_max+1
        self.reg_channels = self.reg_range[1] - self.reg_range[0]
        self.use_varifocal_loss = use_varifocal_loss
        self.assigner = assigner
        self.nms = nms
        if isinstance(self.nms, MultiClassNMS) and trt:
            self.nms.trt = trt
        self.eval_size = eval_size
        self.loss_weight = loss_weight
        self.exclude_nms = exclude_nms
        self.exclude_post_process = exclude_post_process
        self.print_l1_loss = print_l1_loss

        # cls loss
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=paddle.to_tensor([1.0]), reduction="mean")

        # pred head
        c2 = max((16, in_channels[0] // 4, self.reg_channels * 4))
        c3 = max(in_channels[0], self.num_classes)
        c4 = max(in_channels[0] // 4, self.num_masks)
        self.conv_reg = nn.LayerList()
        self.conv_cls = nn.LayerList()
        self.conv_ins = nn.LayerList()
        for in_c in self.in_channels:
            self.conv_reg.append(
                nn.Sequential(* [
                    BaseConv(
                        in_c, c2, 3, 1, act=act),
                    BaseConv(
                        c2, c2, 3, 1, act=act),
                    nn.Conv2D(
                        c2,
                        self.reg_channels * 4,
                        1,
                        bias_attr=ParamAttr(regularizer=L2Decay(0.0))),
                ]))
            self.conv_cls.append(
                nn.Sequential(* [
                    BaseConv(
                        in_c, c3, 3, 1, act=act),
                    BaseConv(
                        c3, c3, 3, 1, act=act),
                    nn.Conv2D(
                        c3,
                        self.num_classes,
                        1,
                        bias_attr=ParamAttr(regularizer=L2Decay(0.0))),
                ]))
            self.conv_ins.append(
                nn.Sequential(* [
                    BaseConv(
                        in_c, c4, 3, 1, act=act),
                    BaseConv(
                        c4, c4, 3, 1, act=act),
                    nn.Conv2D(
                        c4,
                        self.num_masks,
                        1,
                        bias_attr=ParamAttr(regularizer=L2Decay(0.0))),
                ]))
        self.mask_thr_binary = mask_thr_binary
        self.proto = MaskProto(
            in_channels[0], self.num_protos, self.num_masks, act=act)

        # projection conv
        self.dfl_conv = nn.Conv2D(self.reg_channels, 1, 1, bias_attr=False)
        self.dfl_conv.skip_quant = True
        self.proj = paddle.linspace(0, self.reg_channels - 1, self.reg_channels)
        self.dfl_conv.weight.set_value(
            self.proj.reshape([1, self.reg_channels, 1, 1]))
        self.dfl_conv.weight.stop_gradient = True

    def forward(self, feats, targets=None):
        if self.training:
            return self.forward_train(feats, targets)
        else:
            return self.forward_eval(feats)

    def forward_train(self, feats, targets):
        mask_feats = self.proto(feats[0])

        anchors, anchor_points, num_anchors_list, stride_tensor = \
            generate_anchors_for_grid_cell(
                feats, self.fpn_strides, self.grid_cell_scale,
                self.grid_cell_offset)

        cls_score_list, reg_distri_list, mask_coeff_list = [], [], []
        for i, feat in enumerate(feats):
            reg_distri = self.conv_reg[i](feat)
            cls_logit = self.conv_cls[i](feat)
            ins_pred = self.conv_ins[i](feat)

            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.flatten(2).transpose([0, 2, 1]))
            reg_distri_list.append(reg_distri.flatten(2).transpose([0, 2, 1]))
            mask_coeff_list.append(ins_pred.flatten(2).transpose([0, 2, 1]))

        cls_score_list = paddle.concat(cls_score_list, axis=1)
        reg_distri_list = paddle.concat(reg_distri_list, axis=1)
        mask_coeff_list = paddle.concat(mask_coeff_list, axis=1)

        return self.get_loss([
            cls_score_list, reg_distri_list, mask_coeff_list, mask_feats,
            anchors, anchor_points, num_anchors_list, stride_tensor
        ], targets)

    def forward_eval(self, feats):
        mask_feats = self.proto(feats[0])

        anchor_points, stride_tensor = self._generate_anchors(feats)

        cls_score_list, reg_dist_list, mask_coeff_list = [], [], []
        for i, feat in enumerate(feats):
            _, _, h, w = feat.shape
            l = h * w
            reg_dist = self.conv_reg[i](feat)
            cls_logit = self.conv_cls[i](feat)
            mask_coeff = self.conv_ins[i](feat)

            reg_dist = reg_dist.reshape(
                [-1, 4, self.reg_channels, l]).transpose(
                    [0, 2, 3, 1])  # Note diff
            reg_dist = self.dfl_conv(F.softmax(reg_dist, axis=1)).squeeze(1)

            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.reshape([-1, self.num_classes, l]))
            reg_dist_list.append(reg_dist)
            mask_coeff_list.append(mask_coeff.reshape([-1, self.num_masks, l]))

        cls_score_list = paddle.concat(cls_score_list, axis=-1)
        reg_dist_list = paddle.concat(reg_dist_list, axis=1)
        mask_coeff_list = paddle.concat(mask_coeff_list, axis=-1)

        return cls_score_list, reg_dist_list, mask_coeff_list, mask_feats, anchor_points, stride_tensor

    def get_loss(self, head_outs, targets):
        raise NotImplementedError(
            'YOLOv8 InstSeg training not implemented yet.')
        gt_labels = targets['gt_class']
        gt_bboxes = targets['gt_bbox']

        loss_cls = paddle.zeros([1])
        loss_iou = paddle.zeros([1])
        loss = loss_cls + loss_iou
        return {
            'loss': loss,
            'loss_cls': loss_cls,
            'loss_box': loss_iou,
        }

    def post_process(self,
                     head_outs,
                     im_shape,
                     scale_factor,
                     infer_shape=[640, 640],
                     rescale=True):
        assert not self.exclude_post_process or not self.exclude_nms

        pred_scores, pred_dist, pred_mask_coeffs, mask_feats, anchor_points, stride_tensor = head_outs
        pred_bboxes = batch_distance2bbox(anchor_points, pred_dist)
        pred_bboxes *= stride_tensor

        bbox_pred, bbox_num, keep_idxs = self.nms(pred_bboxes, pred_scores)

        if self.with_mask and bbox_num.sum() > 0:
            pred_mask_coeffs = pred_mask_coeffs.transpose([0, 2, 1])  ### Note
            mask_coeffs = paddle.gather(
                pred_mask_coeffs.reshape([-1, self.num_masks]), keep_idxs)

            mask_logits = process_mask_upsample(mask_feats[0], mask_coeffs,
                                                bbox_pred[:, 2:6], infer_shape)
            if rescale:
                ori_h, ori_w = im_shape[0] / scale_factor[0]
                mask_logits = F.interpolate(
                    mask_logits.unsqueeze(0),
                    size=[
                        math.ceil(mask_logits.shape[-2] / scale_factor[0][0]),
                        math.ceil(mask_logits.shape[-1] / scale_factor[0][1])
                    ],
                    mode='bilinear',
                    align_corners=False)[..., :int(ori_h), :int(ori_w)]
            masks = mask_logits.squeeze(0)
            mask_pred = masks > self.mask_thr_binary
        else:
            ori_h, ori_w = im_shape[0] / scale_factor[0]
            mask_pred = paddle.zeros([bbox_num, int(ori_h), int(ori_w)])

        # scale bbox to origin
        scale_factor = scale_factor.flip(-1).tile([1, 2])
        bbox_pred[:, 2:6] /= scale_factor

        if self.with_mask:
            return bbox_pred, bbox_num, mask_pred
        else:
            return bbox_pred, bbox_num


def crop_mask(masks, boxes):
    """
    It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box

    Args:
      masks (paddle.Tensor): [h, w, n] tensor of masks
      boxes (paddle.Tensor): [n, 4] tensor of bbox coordinates in relative point form

    Returns:
      (paddle.Tensor): The masks are being cropped to the bounding box.
    """
    _, h, w = masks.shape
    x1, y1, x2, y2 = paddle.chunk(boxes[:, :, None], 4, axis=1)
    r = paddle.arange(w, dtype=x1.dtype)[None, None, :]
    c = paddle.arange(h, dtype=y1.dtype)[None, :, None]
    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def process_mask_upsample(protos, masks_in, bboxes, shape):
    """
    It takes the output of the mask head, and applies the mask to the bounding boxes. This produces masks of higher
    quality but is slower.

    Args:
      protos (paddle.Tensor): [mask_dim, mask_h, mask_w]
      masks_in (paddle.Tensor): [n, mask_dim], n is number of masks after nms
      bboxes (paddle.Tensor): [n, 4], n is number of masks after nms
      shape (tuple): the size of the input image (h,w)

    Returns:
      (paddle.Tensor): The upsampled masks.
    """
    c, mh, mw = protos.shape  # CHW
    masks = F.sigmoid(masks_in @protos.reshape([c, -1])).reshape([-1, mh, mw])
    masks = F.interpolate(
        masks[None], shape, mode='bilinear', align_corners=False)[0]  # CHW
    masks = crop_mask(masks, bboxes)  # CHW
    return masks
