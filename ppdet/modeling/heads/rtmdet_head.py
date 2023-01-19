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
from ppdet.core.workspace import register

from ..bbox_utils import batch_distance2bbox
from ..losses import GIoULoss, QualityFocalLoss, IouLoss
from ..initializer import bias_init_with_prob, constant_
from ppdet.modeling.backbones.csp_darknet import BaseConv
from ppdet.modeling.assigners.simota_assigner import SimOTAAssigner  #, DynamicSoftLabelAssigner
from ppdet.modeling.layers import MultiClassNMS
from paddle import ParamAttr
from paddle.nn.initializer import Normal

__all__ = ['RTMDetHead']


@register
class RTMDetHead(nn.Layer):
    __shared__ = [
        'num_classes', 'width_mult', 'trt', 'exclude_nms',
        'exclude_post_process'
    ]
    __inject__ = ['assigner', 'nms']

    def __init__(
            self,
            num_classes=80,
            width_mult=1.0,
            in_channels=[1024, 512, 256],
            feat_channels=256,
            stacked_convs=2,
            pred_kernel_size=1,
            act='swish',
            fpn_strides=(32, 16, 8),
            share_conv=True,
            exp_on_reg=False,
            assigner='SimOTAAssigner',  # just placeholder
            grid_cell_offset=0.,
            nms='MultiClassNMS',
            loss_weight={
                'cls': 1.0,
                'box': 2.0,
            },
            trt=False,
            exclude_nms=False,
            exclude_post_process=False):
        super(RTMDetHead, self).__init__()
        assert len(in_channels) > 0, "len(in_channels) should > 0"
        self._dtype = paddle.framework.get_default_dtype()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.pred_kernel_size = pred_kernel_size
        self.stacked_convs = stacked_convs
        self.feat_channels = int(feat_channels * width_mult)
        self.share_conv = share_conv
        self.exp_on_reg = exp_on_reg
        self.grid_cell_offset = grid_cell_offset

        self.loss_cls = QualityFocalLoss()
        self.loss_box = IouLoss(loss_weight=1.0, giou=True)
        self.loss_weight = loss_weight
        self.assigner = assigner

        self.nms = nms
        if isinstance(self.nms, MultiClassNMS) and trt:
            self.nms.trt = trt
        self.exclude_nms = exclude_nms
        self.exclude_post_process = exclude_post_process

        # head
        self.cls_convs = nn.LayerList()
        self.reg_convs = nn.LayerList()
        self.cls_preds = nn.LayerList()
        self.reg_preds = nn.LayerList()
        for idx in range(len(self.fpn_strides)):
            cls_convs = nn.LayerList()
            reg_convs = nn.LayerList()
            for i in range(self.stacked_convs):
                chn = self.in_channels[idx] if i == 0 else self.feat_channels
                cls_convs.append(BaseConv(chn, self.feat_channels, 3, 1))
                reg_convs.append(BaseConv(chn, self.feat_channels, 3, 1))
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)

            self.cls_preds.append(
                nn.Conv2D(
                    self.feat_channels,
                    self.num_classes,
                    self.pred_kernel_size,
                    padding=self.pred_kernel_size // 2,
                    weight_attr=ParamAttr(initializer=Normal(
                        mean=0., std=0.01)),
                    bias_attr=True))
            self.reg_preds.append(
                nn.Conv2D(
                    self.feat_channels,
                    4,
                    self.pred_kernel_size,
                    padding=self.pred_kernel_size // 2,
                    weight_attr=ParamAttr(initializer=Normal(
                        mean=0., std=0.01)),
                    bias_attr=True))

        self.share_conv = False  # TODO in deploy
        if self.share_conv:
            for n in range(len(self.fpn_strides)):
                for i in range(self.stacked_convs):
                    self.cls_convs[n][i].conv = self.cls_convs[0][i].conv
                    self.reg_convs[n][i].conv = self.reg_convs[0][i].conv
        self._init_weights()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def _init_weights(self):
        bias_cls = bias_init_with_prob(0.01)
        for cls_, reg_ in zip(self.cls_preds, self.reg_preds):
            constant_(cls_.weight)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.weight)
            constant_(reg_.bias, 1.0)

    def forward_train(self, feats, targets):
        assert len(feats) == len(self.fpn_strides), \
            "The size of feats is not equal to size of fpn_strides"
        feat_sizes = [[f.shape[-2], f.shape[-1]] for f in feats]

        cls_score_list, reg_distri_list = [], []
        for idx, x in enumerate(feats):
            _, _, h, w = x.shape
            cls_feat = x
            reg_feat = x
            for cls_layer in self.cls_convs[idx]:
                cls_feat = cls_layer(cls_feat)
            cls_logit = self.cls_preds[idx](cls_feat)

            for reg_layer in self.reg_convs[idx]:
                reg_feat = reg_layer(reg_feat)
            if self.exp_on_reg:
                reg_dist = self.reg_preds[idx](reg_feat).exp()
            else:
                reg_dist = self.reg_preds[idx](reg_feat)
            reg_dist = reg_dist * self.fpn_strides[idx]
            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.flatten(2).transpose([0, 2, 1]))
            reg_distri_list.append(reg_dist.flatten(2).transpose([0, 2, 1]))
        cls_score_list = paddle.concat(cls_score_list, axis=1)
        reg_distri_list = paddle.concat(reg_distri_list, axis=1)

        anchor_points, stride_tensor = self._generate_anchor_point(
            feat_sizes, self.fpn_strides, 0.)

        raise NotImplementedError('RTMDet training not implemented yet.')

        return self.get_loss(
            [cls_score_list, reg_distri_list, anchor_points,
             stride_tensor], targets)

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
        for idx, x in enumerate(feats):
            _, _, h, w = x.shape
            l = h * w
            cls_feat = x
            reg_feat = x
            for cls_layer in self.cls_convs[idx]:
                cls_feat = cls_layer(cls_feat)
            cls_logit = self.cls_preds[idx](cls_feat)

            for reg_layer in self.reg_convs[idx]:
                reg_feat = reg_layer(reg_feat)
            if self.exp_on_reg:
                reg_dist = self.reg_preds[idx](reg_feat).exp()
            else:
                reg_dist = self.reg_preds[idx](reg_feat)
            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.reshape([-1, self.num_classes, l]))
            reg_dist_list.append(reg_dist.reshape([-1, 4, l]))

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

    def _generate_anchor_point(self, feat_sizes, strides, offset=0.):
        anchor_points, stride_tensor = [], []
        num_anchors_list = []
        for feat_size, stride in zip(feat_sizes, strides):
            h, w = feat_size
            x = (paddle.arange(w) + offset) * stride
            y = (paddle.arange(h) + offset) * stride
            y, x = paddle.meshgrid(y, x)
            anchor_points.append(paddle.stack([x, y], axis=-1).reshape([-1, 2]))
            stride_tensor.append(
                paddle.full(
                    [len(anchor_points[-1]), 1], stride, dtype=self._dtype))
            num_anchors_list.append(len(anchor_points[-1]))
        anchor_points = paddle.concat(anchor_points).astype(self._dtype)
        anchor_points.stop_gradient = True
        stride_tensor = paddle.concat(stride_tensor)
        stride_tensor.stop_gradient = True
        return anchor_points, stride_tensor  #, num_anchors_list

    def get_loss(self, head_outs, targets):
        pred_cls, pred_bboxes, anchor_points, stride_tensor = head_outs
        raise NotImplementedError('RTMDet training not implemented yet.')

        gt_labels = targets['gt_class']
        gt_bboxes = targets['gt_bbox']

        loss_cls = paddle.zeros([1])
        loss_iou = paddle.zeros([1])
        loss = self.loss_weight['cls'] * loss_cls + \
               self.loss_weight['box'] * loss_iou
        return {
            'loss': loss,
            'loss_cls': loss_cls,
            'loss_box': loss_iou,
        }

    def post_process(self, head_outs, im_shape, scale_factor):
        pred_scores, pred_dist, anchor_points, stride_tensor = head_outs
        pred_bboxes = batch_distance2bbox(anchor_points,
                                          pred_dist.transpose([0, 2, 1]))
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
