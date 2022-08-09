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

import numpy as np
from ..initializer import bias_init_with_prob, constant_

from ..backbones.efficientrep import Conv
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
                 anchors=1,
                 num_layers=3,
                 act='silu',
                 assigner=SimOTAAssigner(use_vfl=False),
                 nms='MultiClassNMS',
                 loss_weight={
                     'cls': 1.0,
                     'obj': 1.0,
                     'iou': 3.0,
                     'l1': 1.0,
                     'reg': 5.0,
                 },
                 iou_type='siou',
                 trt=False,
                 exclude_nms=False):
        super().__init__()
        self._dtype = paddle.framework.get_default_dtype()
        self.num_classes = num_classes
        assert len(in_channels) > 0, "in_channels length should > 0"
        self.in_channels = [int(in_c * width_mult) for in_c in in_channels]
        feat_channels = self.in_channels
        self.fpn_strides = fpn_strides

        self.loss_weight = loss_weight
        self.iou_type = iou_type

        self.assigner = assigner
        self.nms = nms
        if isinstance(self.nms, MultiClassNMS) and trt:
            self.nms.trt = trt
        self.exclude_nms = exclude_nms

        if isinstance(anchors, (list, tuple)):
            self.na = len(anchors[0]) // 2
        else:
            self.na = anchors
        self.anchors = anchors
        self.grid = [paddle.zeros([1])] * num_layers
        self.stride = paddle.to_tensor(fpn_strides)

        ConvBlock = Conv
        self.stem_conv = nn.LayerList()
        self.cls_convs = nn.LayerList()
        self.reg_convs = nn.LayerList()  # reg [x,y,w,h] + obj
        self.cls_preds = nn.LayerList()
        self.reg_preds = nn.LayerList()
        self.obj_preds = nn.LayerList()
        for in_c, feat_c in zip(self.in_channels, feat_channels):
            self.stem_conv.append(Conv(in_c, feat_c, 1, 1))
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
                    4,  # reg [x,y,w,h] + obj
                    1,
                    bias_attr=ParamAttr(regularizer=L2Decay(0.0))))
            self.obj_preds.append(
                nn.Conv2D(
                    feat_c,
                    1,  # reg [x,y,w,h] + obj
                    1,
                    bias_attr=ParamAttr(regularizer=L2Decay(0.0))))
        if self.iou_type == 'ciou':
            self.iou_loss = IouLoss(loss_weight=1.0, ciou=True)
        elif self.iou_type == 'siou':
            self.iou_loss = SIoULoss(splited=False)
        else:
            self.iou_loss = IouLoss(loss_weight=1.0)
        self._initialize_biases()

    def _initialize_biases(self):
        bias_init = bias_init_with_prob(0.01)
        for cls_, obj_ in zip(self.cls_preds, self.obj_preds):
            constant_(cls_.weight)
            constant_(cls_.bias, bias_init)
            constant_(obj_.weight)
            constant_(obj_.bias, bias_init)

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
        return anchor_points, stride_tensor, num_anchors_list

    def forward(self, feats, targets=None):
        assert len(feats) == len(self.fpn_strides), \
            "The size of feats is not equal to size of fpn_strides"

        feat_sizes = [[f.shape[-2], f.shape[-1]] for f in feats]
        cls_score_list, reg_pred_list = [], []
        obj_score_list = []
        for i, feat in enumerate(feats):
            feat = self.stem_conv[i](feat)
            cls_feat = self.cls_convs[i](feat)
            reg_feat = self.reg_convs[i](feat)
            cls_logit = self.cls_preds[i](cls_feat)
            reg_xywh = self.reg_preds[i](reg_feat)
            obj_logit = self.obj_preds[i](reg_feat)
            # cls prediction
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.flatten(2).transpose([0, 2, 1]))
            # reg prediction
            reg_xywh = reg_xywh.flatten(2).transpose([0, 2, 1])
            reg_pred_list.append(reg_xywh)
            # obj prediction
            obj_score = F.sigmoid(obj_logit)
            obj_score_list.append(obj_score.flatten(2).transpose([0, 2, 1]))

        cls_score_list = paddle.concat(cls_score_list, axis=1)
        reg_pred_list = paddle.concat(reg_pred_list, axis=1)
        obj_score_list = paddle.concat(obj_score_list, axis=1)

        # bbox decode
        anchor_points, stride_tensor, _ =\
            self._generate_anchor_point(feat_sizes, self.fpn_strides)
        reg_xy, reg_wh = paddle.split(reg_pred_list, 2, axis=-1)
        reg_xy += (anchor_points / stride_tensor)
        reg_wh = paddle.exp(reg_wh) * 0.5
        bbox_pred_list = paddle.concat(
            [reg_xy - reg_wh, reg_xy + reg_wh], axis=-1)

        if self.training:
            anchor_points, stride_tensor, num_anchors_list =\
                self._generate_anchor_point(feat_sizes, self.fpn_strides, 0.5)
            yolox_losses = self.get_loss([
                cls_score_list, bbox_pred_list, obj_score_list, anchor_points,
                stride_tensor, num_anchors_list
            ], targets)
            return yolox_losses
        else:
            pred_scores = (cls_score_list * obj_score_list).sqrt()
            return pred_scores, bbox_pred_list, stride_tensor

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
        pred_scores, pred_bboxes, stride_tensor = head_outs
        pred_scores = pred_scores.transpose([0, 2, 1])
        pred_bboxes *= stride_tensor
        # scale bbox to origin image
        scale_factor = scale_factor.flip(-1).tile([1, 2]).unsqueeze(1)
        pred_bboxes /= scale_factor
        if self.exclude_nms:
            # `exclude_nms=True` just use in benchmark
            return pred_bboxes.sum(), pred_scores.sum()
        else:
            bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
            return bbox_pred, bbox_num
