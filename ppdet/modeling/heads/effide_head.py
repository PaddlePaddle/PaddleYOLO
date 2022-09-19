# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from IPython import embed
from ..assigners.utils import generate_anchors_for_grid_cell
from ..bbox_utils import batch_distance2bbox
import numpy as np
from ..initializer import bias_init_with_prob, constant_

from ..backbones.efficientrep import BaseConv
from ppdet.modeling.assigners.simota_assigner import SimOTAAssigner
from ppdet.modeling.bbox_utils import bbox_overlaps
from ..losses import IouLoss, SIoULoss
from ppdet.modeling.layers import MultiClassNMS
from ppdet.core.workspace import register, serializable
from ..shape_spec import ShapeSpec

__all__ = ['EffiDeHead']


@register
@serializable
class EffiDeHead(nn.Layer):
    __shared__ = [
        'num_classes', 'width_mult', 'depth_mult', 'act', 'trt', 'exclude_nms'
    ]
    __inject__ = ['assigner', 'nms']

    def __init__(self,
                 num_classes=80,
                 depth_mult=1.0,
                 width_mult=1.0,
                 depthwise=False,
                 in_channels=[128, 256, 512],
                 fpn_strides=[8, 16, 32],
                 act='relu',
                 assigner=SimOTAAssigner(use_vfl=False),
                 nms='MultiClassNMS',
                 loss_weight={
                     'cls': 1.0,
                     'obj': 1.0,
                     'iou': 3.0,
                     'l1': 1.0,
                     'reg': 5.0,
                 },
                 distill_weight={
                     'class': 0.0,
                     'dfl': 0.0,
                 },
                 num_anchors=1,
                 grid_cell_scale=5.0,
                 grid_cell_offset=0.5,
                 reg_max=0,
                 use_dfl=False,
                 iou_type='siou',
                 trt=False,
                 exclude_nms=False,
                 exclude_post_process=False):
        super(EffiDeHead, self).__init__()
        self._dtype = paddle.framework.get_default_dtype()
        self.num_classes = num_classes
        assert len(in_channels) > 0, "in_channels length should > 0"
        self.in_channels = [int(in_c * width_mult) for in_c in in_channels]
        feat_channels = self.in_channels
        self.fpn_strides = fpn_strides
        self.act = act
        self.assigner = assigner
        self.nms = nms
        if isinstance(self.nms, MultiClassNMS) and trt:
            self.nms.trt = trt
        self.loss_weight = loss_weight
        self.iou_type = iou_type
        self.exclude_nms = exclude_nms
        self.exclude_post_process = exclude_post_process

        ConvBlock = BaseConv  # todo: depthwise
        self.stem_conv = nn.LayerList()
        self.cls_convs = nn.LayerList()
        self.reg_convs = nn.LayerList()
        self.cls_preds = nn.LayerList()
        self.reg_preds = nn.LayerList()

        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset

        self.reg_max = reg_max
        self.use_dfl = use_dfl
        self.num_anchors = num_anchors

        for in_c, feat_c in zip(self.in_channels, feat_channels):
            self.stem_conv.append(BaseConv(in_c, feat_c, 1, 1))
            self.cls_convs.append(ConvBlock(feat_c, feat_c, 3, 1))
            self.reg_convs.append(ConvBlock(feat_c, feat_c, 3, 1))

            self.cls_preds.append(
                nn.Conv2D(
                    feat_c,
                    self.num_classes,
                    1,
                    bias_attr=ParamAttr(regularizer=L2Decay(0.0))))
            self.reg_preds.append(
                nn.Conv2D(
                    feat_c,
                    4 * (self.reg_max + self.num_anchors),  # reg [x,y,w,h]
                    1,
                    bias_attr=ParamAttr(regularizer=L2Decay(0.0))))

        if self.iou_type == 'ciou':
            self.iou_loss = IouLoss(loss_weight=1.0, ciou=True)
        elif self.iou_type == 'siou':
            self.iou_loss = SIoULoss(splited=False)
        elif self.iou_type == 'giou':
            self.iou_loss = SIoULoss(splited=False)
        else:
            self.iou_loss = IouLoss(loss_weight=1.0)
        # self._initialize_biases()

        # projection conv
        if self.use_dfl:
            self.proj_conv = nn.Conv2D(self.reg_max + 1, 1, 1, bias_attr=False)
            self.proj_conv.skip_quant = True
        self._init_weights()

    def _initialize_biases(self):
        bias_init = bias_init_with_prob(0.01)
        for cls_, obj_ in zip(self.cls_preds, self.obj_preds):
            constant_(cls_.weight)
            constant_(cls_.bias, bias_init)
            constant_(obj_.weight)
            constant_(obj_.bias, bias_init)

    def _init_weights(self):
        bias_cls = bias_init_with_prob(0.01)
        for cls_, reg_ in zip(self.cls_preds, self.reg_preds):
            constant_(cls_.weight)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.weight)
            constant_(reg_.bias, 1.0)

        # projection conv
        if self.use_dfl:
            proj = paddle.linspace(0, self.reg_max, self.reg_max + 1).reshape(
                [1, self.reg_max + 1, 1, 1])
            self.proj_conv.weight.set_value(proj)
            self.proj_conv.weight.stop_gradient = True

    def forward_train(self, feats, targets):
        anchors, anchor_points, num_anchors_list, stride_tensor = \
            generate_anchors_for_grid_cell(
                feats, self.fpn_strides, self.grid_cell_scale,
                self.grid_cell_offset)

        cls_score_list, reg_distri_list = [], []
        for i, feat in enumerate(feats):
            feat = self.stem_conv[i](feat)
            cls_logit = self.cls_preds[i](self.cls_convs[i](feat))
            reg_distri = self.reg_preds[i](self.reg_convs[i](feat))
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

    def forward_eval(self, feats):
        anchor_points, stride_tensor = self._generate_anchors(feats)

        cls_score_list, reg_dist_list = [], []
        for i, feat in enumerate(feats):
            b, _, h, w = feat.shape  # [1, 64, 80, 80] [1, 128, 40, 40] [1, 256, 20, 20]
            l = h * w
            feat = self.stem_conv[i](feat)
            cls_logit = self.cls_preds[i](self.cls_convs[i](feat))
            reg_dist = self.reg_preds[i](
                self.reg_convs[i](feat))  # [1, 4, 80, 80]

            if self.use_dfl:
                reg_dist = reg_dist.reshape(
                    [-1, 4, self.reg_max + 1, l]).transpose([0, 2, 1, 3])
                reg_dist = self.proj_conv(F.softmax(reg_dist, axis=1))
            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.reshape([b, self.num_classes, l]))
            reg_dist_list.append(reg_dist.reshape([b, 4, l]))

        cls_score_list = paddle.concat(cls_score_list, axis=-1)
        reg_dist_list = paddle.concat(reg_dist_list, axis=-1)

        return cls_score_list, reg_dist_list, anchor_points, stride_tensor

    def forward(self, feats, targets=None):
        assert len(feats) == len(self.fpn_strides), \
            "The size of feats is not equal to size of fpn_strides"

        if self.training:
            return self.forward_train(feats, targets)
        else:
            return self.forward_eval(feats)

    def get_loss(self, head_outs, targets):
        pred_cls, pred_bboxes, pred_obj,\
        anchor_points, stride_tensor, num_anchors_list = head_outs
        gt_labels = targets['gt_class']
        gt_bboxes = targets['gt_bbox']

        pred_scores = (pred_cls * pred_obj).sqrt()
        # label assignment
        center_and_strides = paddle.concat(
            [anchor_points, stride_tensor, stride_tensor], axis=-1)
        pos_num_list, label_list, bbox_target_list = [], [], []
        for pred_score, pred_bbox, gt_box, gt_label in zip(
                pred_scores.detach(),
                pred_bboxes.detach() * stride_tensor, gt_bboxes, gt_labels):
            gt_box = paddle.cast(gt_box, 'float32')
            gt_label = paddle.cast(gt_label, 'float32')
            pos_num, label, _, bbox_target = self.assigner(
                pred_score, center_and_strides, pred_bbox, gt_box, gt_label)
            pos_num_list.append(pos_num)
            label_list.append(label)
            bbox_target_list.append(bbox_target)
        labels = paddle.to_tensor(np.stack(label_list, axis=0))
        bbox_targets = paddle.to_tensor(np.stack(bbox_target_list, axis=0))
        bbox_targets /= stride_tensor  # rescale bbox

        # 1. obj score loss
        mask_positive = (labels != self.num_classes)
        loss_obj = F.binary_cross_entropy(
            pred_obj,
            mask_positive.astype(pred_obj.dtype).unsqueeze(-1),
            reduction='sum')

        num_pos = sum(pos_num_list)

        if num_pos > 0:
            num_pos = paddle.to_tensor(num_pos, dtype=self._dtype).clip(min=1)
            loss_obj /= num_pos

            # 2. iou loss
            bbox_mask = mask_positive.unsqueeze(-1).tile([1, 1, 4])
            pred_bboxes_pos = paddle.masked_select(pred_bboxes,
                                                   bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = paddle.masked_select(
                bbox_targets, bbox_mask).reshape([-1, 4])
            bbox_iou = bbox_overlaps(pred_bboxes_pos, assigned_bboxes_pos)
            bbox_iou = paddle.diag(bbox_iou)

            loss_iou = self.iou_loss(
                pred_bboxes_pos.split(
                    4, axis=-1),
                assigned_bboxes_pos.split(
                    4, axis=-1))
            loss_iou = loss_iou.sum() / num_pos

            # 3. cls loss
            cls_mask = mask_positive.unsqueeze(-1).tile(
                [1, 1, self.num_classes])
            pred_cls_pos = paddle.masked_select(
                pred_cls, cls_mask).reshape([-1, self.num_classes])
            assigned_cls_pos = paddle.masked_select(labels, mask_positive)
            assigned_cls_pos = F.one_hot(assigned_cls_pos,
                                         self.num_classes + 1)[..., :-1]
            assigned_cls_pos *= bbox_iou.unsqueeze(-1)
            loss_cls = F.binary_cross_entropy(
                pred_cls_pos, assigned_cls_pos, reduction='sum')
            loss_cls /= num_pos

            # 4. l1 loss
            loss_l1 = F.l1_loss(
                pred_bboxes_pos, assigned_bboxes_pos, reduction='sum')
            loss_l1 /= num_pos
        else:
            loss_cls = paddle.zeros([1])
            loss_iou = paddle.zeros([1])
            loss_l1 = paddle.zeros([1])
            loss_cls.stop_gradient = False
            loss_iou.stop_gradient = False
            loss_l1.stop_gradient = False

        loss = self.loss_weight['obj'] * loss_obj + \
               self.loss_weight['cls'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou * self.loss_weight['reg'] + \
               self.loss_weight['l1'] * loss_l1

        yolox_losses = {
            'loss': loss,
            'loss_cls': loss_cls,
            'loss_obj': loss_obj,
            'loss_iou': loss_iou,
            'loss_l1': loss_l1,
        }
        return yolox_losses

    def post_process(self, head_outs, img_shape, scale_factor):
        pred_scores, pred_dist, anchor_points, stride_tensor = head_outs
        pred_bboxes = batch_distance2bbox(anchor_points,
                                          pred_dist.transpose(
                                              [0, 2, 1]))  #, box_format='xywh')
        pred_bboxes *= stride_tensor

        if self.exclude_post_process:
            return paddle.concat(
                [pred_bboxes, pred_scores.transpose([0, 2, 1])], axis=-1), None
        else:
            # scale bbox to origin
            scale_y, scale_x = paddle.split(scale_factor, 2, axis=-1)
            scale_factor = paddle.concat(
                [scale_x, scale_y, scale_x, scale_y],
                axis=-1).reshape([-1, 1, 4])
            pred_bboxes /= scale_factor
            if self.exclude_nms:
                # `exclude_nms=True` just use in benchmark
                return pred_bboxes, pred_scores
            else:
                bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
                return bbox_pred, bbox_num
