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
from ppdet.core.workspace import register

from ..bbox_utils import batch_distance2bbox
from ..losses import GIoULoss, SIoULoss
from ..initializer import bias_init_with_prob, constant_, normal_
from ..assigners.utils import generate_anchors_for_grid_cell
from ..backbones.yolov6_efficientrep import BaseConv
from ppdet.modeling.layers import MultiClassNMS
from ppdet.modeling.assigners.task_aligned_assigner import TaskAlignedAssigner

__all__ = ['EffiDeHead', 'EffiDeHead_distill_ns', 'EffiDeHead_fuseab']


@register
class EffiDeHead(nn.Layer):
    __shared__ = [
        'num_classes', 'eval_size', 'trt', 'exclude_nms',
        'exclude_post_process', 'self_distill'
    ]
    __inject__ = ['static_assigner', 'assigner', 'nms']

    def __init__(
            self,
            in_channels=[128, 256, 512],
            num_classes=80,
            fpn_strides=[8, 16, 32],
            grid_cell_scale=5.0,
            grid_cell_offset=0.5,
            anchors=1,
            reg_max=16,  # reg_max=0 if use_dfl is False
            use_dfl=True,  # False in n/s version, True in m/l version
            static_assigner_epoch=4,  # warmup_epoch
            static_assigner='ATSSAssigner',
            assigner='TaskAlignedAssigner',
            eval_size=None,
            iou_type='giou',  # 'siou' in n version
            loss_weight={
                'cls': 1.0,
                'iou': 2.5,
                'dfl': 0.5,  # used in m/l version 
                'cwd': 10.0,  # used when self_distill=True, in m/l version
            },
            nms='MultiClassNMS',
            trt=False,
            exclude_nms=False,
            exclude_post_process=False,
            self_distill=False,
            distill_weight={
                'cls': 1.0,
                'dfl': 1.0,
            },
            print_l1_loss=True):
        super(EffiDeHead, self).__init__()
        assert len(in_channels) > 0, "len(in_channels) should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.reg_max = reg_max
        self.use_dfl = use_dfl

        if isinstance(anchors, (list, tuple)):
            self.na = len(anchors[0]) // 2
        else:
            self.na = anchors
        self.anchors = anchors

        self.static_assigner_epoch = static_assigner_epoch
        self.static_assigner = static_assigner
        self.assigner = assigner
        self.eval_size = eval_size
        self.iou_loss = GIoULoss()
        assert iou_type in ['giou', 'siou'], "only support giou and siou loss."
        if iou_type == 'siou':
            self.iou_loss = SIoULoss()
        self.loss_weight = loss_weight

        self.nms = nms
        if isinstance(self.nms, MultiClassNMS) and trt:
            self.nms.trt = trt
        self.exclude_nms = exclude_nms
        self.exclude_post_process = exclude_post_process
        self.print_l1_loss = print_l1_loss

        # for self-distillation
        self.self_distill = self_distill
        self.distill_weight = distill_weight

        # Init decouple head
        self.stems = nn.LayerList()
        self.cls_convs = nn.LayerList()
        self.cls_preds = nn.LayerList()
        self.reg_convs = nn.LayerList()
        self.reg_preds = nn.LayerList()

        bias_attr = ParamAttr(regularizer=L2Decay(0.0))
        reg_ch = self.reg_max + self.na
        cls_ch = self.num_classes * self.na
        for in_c in self.in_channels:
            self.stems.append(BaseConv(in_c, in_c, 1, 1))

            self.cls_convs.append(BaseConv(in_c, in_c, 3, 1))
            self.cls_preds.append(
                nn.Conv2D(
                    in_c, cls_ch, 1, bias_attr=bias_attr))

            self.reg_convs.append(BaseConv(in_c, in_c, 3, 1))
            self.reg_preds.append(
                nn.Conv2D(
                    in_c, 4 * reg_ch, 1, bias_attr=bias_attr))

        self.use_dfl = use_dfl
        self.reg_max = reg_max
        self.proj_conv = nn.Conv2D(self.reg_max + 1, 1, 1, bias_attr=False)
        self.proj_conv.skip_quant = True

        self.proj = paddle.linspace(0, self.reg_max, self.reg_max + 1)
        self.proj_conv.weight.set_value(
            self.proj.reshape([1, self.reg_max + 1, 1, 1]))
        self.proj_conv.weight.stop_gradient = True

        # self._init_weights()
        self.print_l1_loss = print_l1_loss

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

        self.proj = paddle.linspace(0, self.reg_max, self.reg_max + 1)
        self.proj_conv.weight.set_value(
            self.proj.reshape([1, self.reg_max + 1, 1, 1]))
        self.proj_conv.weight.stop_gradient = True

        if self.eval_size:
            anchor_points, stride_tensor = self._generate_anchors()
            self.anchor_points = anchor_points
            self.stride_tensor = stride_tensor

    def forward(self, feats, targets=None):
        assert len(feats) == len(self.fpn_strides)
        if self.training:
            return self.forward_train(feats, targets)
        else:
            return self.forward_eval(feats)

    def forward_train(self, feats, targets):
        cls_score_list, reg_distri_list = [], []
        for i, feat in enumerate(feats):
            feat = self.stems[i](feat)
            cls_x = feat
            reg_x = feat
            cls_feat = self.cls_convs[i](cls_x)
            cls_output = self.cls_preds[i](cls_feat)
            reg_feat = self.reg_convs[i](reg_x)
            reg_output = self.reg_preds[i](reg_feat)
            # cls and reg
            cls_output = F.sigmoid(cls_output)
            cls_score_list.append(paddle.transpose(cls_output.flatten(2), perm=[0, 2, 1]))
            reg_distri_list.append(paddle.transpose(reg_output.flatten(2), perm=[0, 2, 1]))

        cls_score_list = paddle.concat(cls_score_list, axis=1)
        reg_distri_list = paddle.concat(reg_distri_list, axis=1)

        return [feats, cls_score_list, reg_distri_list, targets]

    def forward_eval(self, feats):
        anchor_points, stride_tensor = self._generate_anchors(feats)
        cls_score_list, reg_dist_list = [], []
        for i, feat in enumerate(feats):
            _, _, h, w = feat.shape
            l = h * w
            feat = self.stems[i](feat)
            cls_x = feat
            reg_x = feat
            cls_feat = self.cls_convs[i](cls_x)
            cls_output = self.cls_preds[i](cls_feat)
            reg_feat = self.reg_convs[i](reg_x)
            reg_output = self.reg_preds[i](reg_feat)

            if self.use_dfl:
                reg_output = reg_output.reshape(
                    [-1, 4, self.reg_max + 1, l]).transpose([0, 2, 1, 3])
                reg_output = self.proj_conv(F.softmax(reg_output, 1))

            # cls and reg
            cls_output = F.sigmoid(cls_output)
            cls_score_list.append(cls_output.reshape([-1, self.num_classes, l]))
            reg_dist_list.append(reg_output.reshape([-1, 4, l]))

        cls_score_list = paddle.concat(cls_score_list, axis=-1)
        reg_dist_list = paddle.concat(reg_dist_list, axis=-1)

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


@register
class EffiDeHead_distill_ns(EffiDeHead):
    # add reg_preds_lrtb
    __shared__ = [
        'num_classes', 'eval_size', 'trt', 'exclude_nms',
        'exclude_post_process', 'self_distill'
    ]
    __inject__ = ['static_assigner', 'assigner', 'nms']

    def __init__(
            self,
            in_channels=[128, 256, 512],
            num_classes=80,
            fpn_strides=[8, 16, 32],
            grid_cell_scale=5.0,
            grid_cell_offset=0.5,
            anchors=1,
            reg_max=16,  # reg_max=0 if use_dfl is False
            use_dfl=True,  # False in n/s version, True in m/l version
            static_assigner_epoch=4,  # warmup_epoch
            static_assigner='ATSSAssigner',
            assigner='TaskAlignedAssigner',
            eval_size=None,
            iou_type='giou',  # 'siou' in n version
            loss_weight={
                'cls': 1.0,
                'iou': 2.5,
                'dfl': 0.5,  # used in m/l version 
                'cwd': 10.0,  # used when self_distill=True, in m/l version
            },
            nms='MultiClassNMS',
            trt=False,
            exclude_nms=False,
            exclude_post_process=False,
            self_distill=False,
            distill_weight={
                'cls': 1.0,
                'dfl': 1.0,
            },
            print_l1_loss=True):
        super(EffiDeHead_distill_ns, self).__init__()
        assert len(in_channels) > 0, "len(in_channels) should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.reg_max = reg_max
        self.use_dfl = use_dfl

        if isinstance(anchors, (list, tuple)):
            self.na = len(anchors[0]) // 2
        else:
            self.na = anchors
        self.anchors = anchors

        self.static_assigner_epoch = static_assigner_epoch
        self.static_assigner = static_assigner
        self.assigner = assigner
        self.eval_size = eval_size
        self.iou_loss = GIoULoss()
        assert iou_type in ['giou', 'siou'], "only support giou and siou loss."
        if iou_type == 'siou':
            self.iou_loss = SIoULoss()
        self.loss_weight = loss_weight

        self.nms = nms
        if isinstance(self.nms, MultiClassNMS) and trt:
            self.nms.trt = trt
        self.exclude_nms = exclude_nms
        self.exclude_post_process = exclude_post_process
        self.print_l1_loss = print_l1_loss

        # for self-distillation
        self.self_distill = self_distill
        self.distill_weight = distill_weight

        # Init decouple head
        self.stems = nn.LayerList()
        self.cls_convs = nn.LayerList()
        self.cls_preds = nn.LayerList()
        self.reg_convs = nn.LayerList()
        self.reg_preds = nn.LayerList()
        self.reg_preds_lrtb = nn.LayerList()

        bias_attr = ParamAttr(regularizer=L2Decay(0.0))
        reg_ch = self.reg_max + self.na
        for i, in_c in enumerate(self.in_channels):
            self.stems.append(BaseConv(in_c, in_c, 1, 1))
            self.cls_convs.append(BaseConv(in_c, in_c, 3, 1))
            self.cls_preds.append(
                nn.Conv2D(
                    in_c, self.num_classes * self.na, 1, bias_attr=bias_attr))
            self.reg_convs.append(BaseConv(in_c, in_c, 3, 1))
            self.reg_preds.append(
                nn.Conv2D(
                    in_c, 4 * reg_ch, 1, bias_attr=bias_attr))
            self.reg_preds_lrtb.append(
                nn.Conv2D(
                    in_c, 4 * self.na, 1, bias_attr=bias_attr))

        self.use_dfl = use_dfl
        self.reg_max = reg_max
        self.proj_conv = nn.Conv2D(self.reg_max + 1, 1, 1, bias_attr=False)
        self.proj_conv.skip_quant = True

        self.proj = paddle.linspace(0, self.reg_max, self.reg_max + 1)
        self.proj_conv.weight.set_value(
            self.proj.reshape([1, self.reg_max + 1, 1, 1]))
        self.proj_conv.weight.stop_gradient = True

        self._init_weights()
        self.print_l1_loss = print_l1_loss

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
        for reg_pred in self.reg_preds:
            constant_(reg_pred.weight)
            constant_(reg_pred.bias, 1.0)

        self.proj = paddle.linspace(0, self.reg_max, self.reg_max + 1)
        self.proj_conv.weight.set_value(
            self.proj.reshape([1, self.reg_max + 1, 1, 1]))
        self.proj_conv.weight.stop_gradient = True

        if self.eval_size:
            anchor_points, stride_tensor = self._generate_anchors()
            self.anchor_points = anchor_points
            self.stride_tensor = stride_tensor

    def forward(self, feats, targets=None):
        assert len(feats) == len(self.fpn_strides)
        if self.training:
            return self.forward_train(feats, targets)
        else:
            return self.forward_eval(feats)

    def forward_train(self, feats, targets):
        cls_score_list, reg_distri_list, reg_lrtb_list = [], [], []
        for i, feat in enumerate(feats):
            feat = self.stems[i](feat)
            cls_x = feat
            reg_x = feat
            cls_feat = self.cls_convs[i](cls_x)
            cls_output = self.cls_preds[i](cls_feat)
            reg_feat = self.reg_convs[i](reg_x)
            reg_output = self.reg_preds[i](reg_feat)
            reg_output_lrtb = self.reg_preds_lrtb[i](reg_feat)
            # cls and reg
            cls_output = F.sigmoid(cls_output)

            cls_score_list.append(paddle.transpose(cls_output.flatten(2), perm=[0, 2, 1]))
            reg_distri_list.append(paddle.transpose(reg_output.flatten(2), perm=[0, 2, 1]))
            reg_lrtb_list.append(paddle.transpose(reg_output_lrtb.flatten(2), perm=[0, 2, 1]))

        cls_score_list = paddle.concat(cls_score_list, axis=1)
        reg_distri_list = paddle.concat(reg_distri_list, axis=1)
        reg_lrtb_list = paddle.concat(reg_lrtb_list, axis=1)

        return [feats, cls_score_list, reg_distri_list, reg_lrtb_list, targets]

    def forward_eval(self, feats):
        anchor_points, stride_tensor = self._generate_anchors(feats)
        cls_score_list, reg_lrtb_list = [], []
        for i, feat in enumerate(feats):
            _, _, h, w = feat.shape
            l = h * w
            feat = self.stems[i](feat)
            cls_x = feat
            reg_x = feat
            cls_feat = self.cls_convs[i](cls_x)
            cls_output = self.cls_preds[i](cls_feat)
            reg_feat = self.reg_convs[i](reg_x)
            reg_output_lrtb = self.reg_preds_lrtb[i](reg_feat)
            # cls and reg_lrtb 
            cls_output = F.sigmoid(cls_output)
            cls_score_list.append(cls_output.reshape([-1, self.num_classes, l]))
            reg_lrtb_list.append(reg_output_lrtb.reshape([-1, 4, l]))

        cls_score_list = paddle.concat(cls_score_list, axis=-1)
        reg_lrtb_list = paddle.concat(reg_lrtb_list, axis=-1)

        return cls_score_list, reg_lrtb_list, anchor_points, stride_tensor


@register
class EffiDeHead_fuseab(EffiDeHead):
    # add cls_preds_af/reg_preds_af and cls_preds_ab/reg_preds_ab
    __shared__ = [
        'num_classes', 'eval_size', 'trt', 'exclude_nms',
        'exclude_post_process', 'self_distill'
    ]
    __inject__ = ['static_assigner', 'assigner', 'nms']

    def __init__(
            self,
            in_channels=[128, 256, 512],
            num_classes=80,
            fpn_strides=[8, 16, 32],
            grid_cell_scale=5.0,
            grid_cell_offset=0.5,
            anchors=1,
            reg_max=16,  # reg_max=0 if use_dfl is False
            use_dfl=True,  # False in n/s version, True in m/l version
            static_assigner_epoch=4,  # warmup_epoch
            static_assigner='ATSSAssigner',
            assigner='TaskAlignedAssigner',
            eval_size=None,
            iou_type='giou',  # 'siou' in n version
            loss_weight={
                'cls': 1.0,
                'iou': 2.5,
                'dfl': 0.5,  # used in m/l version
                'cwd': 10.0,  # used when self_distill=True, in m/l version
            },
            nms='MultiClassNMS',
            trt=False,
            exclude_nms=False,
            exclude_post_process=False,
            self_distill=False,
            distill_weight={
                'cls': 1.0,
                'dfl': 1.0,
            },
            print_l1_loss=True):
        super(EffiDeHead_fuseab, self).__init__()
        assert len(in_channels) > 0, "len(in_channels) should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.reg_max = reg_max
        self.use_dfl = use_dfl

        if isinstance(anchors, (list, tuple)):
            self.na = len(anchors[0]) // 2
        else:
            self.na = anchors
        self.anchors = anchors

        self.static_assigner_epoch = static_assigner_epoch
        self.static_assigner = static_assigner
        self.assigner = assigner
        self.assigner_ab = TaskAlignedAssigner(topk=26, alpha=1.0, beta=6.0)
        self.eval_size = eval_size
        self.iou_loss = GIoULoss()
        assert iou_type in ['giou', 'siou'], "only support giou and siou loss."
        if iou_type == 'siou':
            self.iou_loss = SIoULoss()
        self.loss_weight = loss_weight

        self.nms = nms
        if isinstance(self.nms, MultiClassNMS) and trt:
            self.nms.trt = trt
        self.exclude_nms = exclude_nms
        self.exclude_post_process = exclude_post_process
        self.print_l1_loss = print_l1_loss

        # for self-distillation
        self.self_distill = self_distill
        self.distill_weight = distill_weight

        self.anchors_init = (paddle.to_tensor(anchors) / paddle.to_tensor(self.fpn_strides)[:, None]).reshape(
            [3, self.na, 2])
        # Init decouple head
        self.stems = nn.LayerList()
        self.cls_convs = nn.LayerList()
        self.reg_convs = nn.LayerList()
        self.cls_preds_af = nn.LayerList()
        self.reg_preds_af = nn.LayerList()
        self.cls_preds_ab = nn.LayerList()
        self.reg_preds_ab = nn.LayerList()

        bias_attr = ParamAttr(regularizer=L2Decay(0.0))
        reg_ch = self.reg_max + 1
        cls_ch = self.num_classes * self.na
        for in_c in self.in_channels:
            self.stems.append(BaseConv(in_c, in_c, 1, 1))

            self.cls_convs.append(BaseConv(in_c, in_c, 3, 1))
            self.cls_preds_af.append(
                nn.Conv2D(
                    in_c, self.num_classes, 1, bias_attr=bias_attr))
            self.cls_preds_ab.append(
                nn.Conv2D(
                    in_c, cls_ch, 1, bias_attr=bias_attr))

            self.reg_convs.append(BaseConv(in_c, in_c, 3, 1))
            self.reg_preds_af.append(
                nn.Conv2D(
                    in_c, 4 * reg_ch, 1, bias_attr=bias_attr))
            self.reg_preds_ab.append(
                nn.Conv2D(
                    in_c, 4 * self.na, 1, bias_attr=bias_attr))

        self.use_dfl = use_dfl
        self.reg_max = reg_max
        self.proj_conv = nn.Conv2D(self.reg_max + 1, 1, 1, bias_attr=False)
        self.proj_conv.skip_quant = True

        self.proj = paddle.linspace(0, self.reg_max, self.reg_max + 1)
        self.proj_conv.weight.set_value(
            self.proj.reshape([1, self.reg_max + 1, 1, 1]))
        self.proj_conv.weight.stop_gradient = True

        self._init_weights()
        self.print_l1_loss = print_l1_loss

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def _init_weights(self):
        bias_cls = bias_init_with_prob(0.01)
        for cls_, reg_ in zip(self.cls_preds_af, self.reg_preds_af):
            constant_(cls_.weight)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.weight)
            constant_(reg_.bias, 1.0)

        for cls_, reg_ in zip(self.cls_preds_ab, self.reg_preds_ab):
            constant_(cls_.weight)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.weight)
            constant_(reg_.bias, 1.0)

        self.proj = paddle.linspace(0, self.reg_max, self.reg_max + 1)
        self.proj_conv.weight.set_value(
            self.proj.reshape([1, self.reg_max + 1, 1, 1]))
        self.proj_conv.weight.stop_gradient = True

        if self.eval_size:
            anchor_points, stride_tensor = self._generate_anchors()
            self.anchor_points = anchor_points
            self.stride_tensor = stride_tensor

    def forward(self, feats, targets=None):
        assert len(feats) == len(self.fpn_strides)
        if self.training:
            return self.forward_train(feats, targets)
        else:
            return self.forward_eval(feats)

    def forward_train(self, feats, targets):
        anchors, anchor_points, num_anchors_list, stride_tensor = \
            generate_anchors_for_grid_cell(
                feats, self.fpn_strides, self.grid_cell_scale,
                self.grid_cell_offset)

        cls_score_list_af, reg_distri_list_af, cls_score_list_ab, reg_distri_list_ab = [], [], [], []

        for i, feat in enumerate(feats):
            b, _, h, w = feat.shape
            feat = self.stems[i](feat)
            cls_x = feat
            reg_x = feat

            cls_feat = self.cls_convs[i](cls_x)
            reg_feat = self.reg_convs[i](reg_x)

            # anchor_base
            cls_output_ab = self.cls_preds_ab[i](cls_feat)
            reg_output_ab = self.reg_preds_ab[i](reg_feat)

            cls_output_ab = F.sigmoid(cls_output_ab)
            cls_output_ab = cls_output_ab.reshape([b, self.na, -1, h, w])
            cls_output_ab = paddle.transpose(cls_output_ab, perm=[0, 1, 3, 4, 2])
            cls_score_list_ab.append(cls_output_ab.flatten(1, 3))

            reg_output_ab = reg_output_ab.reshape([b, self.na, -1, h, w])
            reg_output_ab = paddle.transpose(reg_output_ab, perm=[0, 1, 3, 4, 2])
            reg_output_ab[..., 2:4] = ((F.sigmoid(reg_output_ab[..., 2:4]) * 2) ** 2) * (
                self.anchors_init[i].reshape([1, self.na, 1, 1, 2]))
            reg_distri_list_ab.append(reg_output_ab.flatten(1, 3))
            # anchor_free
            cls_output_af = self.cls_preds_af[i](cls_feat)
            reg_output_af = self.reg_preds_af[i](reg_feat)
            cls_output_af = F.sigmoid(cls_output_af)
            cls_score_list_af.append(paddle.transpose(cls_output_af.flatten(2), perm=[0, 2, 1]))
            reg_distri_list_af.append(paddle.transpose(reg_output_af.flatten(2), perm=[0, 2, 1]))

        cls_score_list_ab = paddle.concat(cls_score_list_ab, axis=1)
        reg_distri_list_ab = paddle.concat(reg_distri_list_ab, axis=1)
        cls_score_list_af = paddle.concat(cls_score_list_af, axis=1)
        reg_distri_list_af = paddle.concat(reg_distri_list_af, axis=1)

        return [feats, cls_score_list_ab, reg_distri_list_ab, cls_score_list_af, reg_distri_list_af, targets]

    def forward_eval(self, feats):
        anchor_points_af, stride_tensor_af = self._generate_anchors(feats)
        cls_score_list_af, reg_dist_list_af = [], []
        for i, feat in enumerate(feats):
            b, _, h, w = feat.shape
            l = h * w
            feat = self.stems[i](feat)
            cls_x = feat
            reg_x = feat
            cls_feat = self.cls_convs[i](cls_x)
            cls_output_af = self.cls_preds_af[i](cls_feat)
            reg_feat = self.reg_convs[i](reg_x)
            reg_output_af = self.reg_preds_af[i](reg_feat)

            if self.use_dfl:
                reg_output_af = reg_output_af.reshape([-1, 4, self.reg_max + 1, l]).transpose([0, 2, 1, 3])
                reg_output_af = self.proj_conv(F.softmax(reg_output_af, axis=1))

            cls_output_af = F.sigmoid(cls_output_af)
            cls_score_list_af.append(cls_output_af.reshape([b, self.num_classes, l]))
            reg_dist_list_af.append(reg_output_af.reshape([b, 4, l]))

        cls_score_list_af = paddle.concat(cls_score_list_af, axis=-1)
        reg_dist_list_af = paddle.concat(reg_dist_list_af, axis=-1)

        return [cls_score_list_af, reg_dist_list_af, anchor_points_af, stride_tensor_af]

