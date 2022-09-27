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
from ..losses import GIoULoss
from ..initializer import bias_init_with_prob, constant_, normal_
from ..assigners.utils import generate_anchors_for_grid_cell
from ppdet.modeling.backbones.cspresnet import ConvBNLayer
from ppdet.modeling.ops import get_static_shape, get_act_fn
from ppdet.modeling.layers import MultiClassNMS
from paddle import ParamAttr
from paddle.nn.initializer import Normal, Constant

__all__ = ['RTMDetHead']


@register
class RTMDetHead(nn.Layer):
    __shared__ = [
        'num_classes', 'width_mult', 'trt', 'exclude_nms',
        'exclude_post_process'
    ]
    __inject__ = ['static_assigner', 'nms']

    def __init__(self,
                 num_classes=80,
                 width_mult=1.0,
                 in_channels=[1024, 512, 256],
                 feat_channels=256,
                 stacked_convs=2,
                 pred_kernel_size=1,
                 act='swish',
                 fpn_strides=(32, 16, 8),
                 share_conv=True,
                 use_varifocal_loss=True,
                 exp_on_reg=False,
                 static_assigner='ATSSAssigner',
                 nms='MultiClassNMS',
                 loss_weight={
                     'class': 1.0,
                     'iou': 2.5,
                     'dfl': 0.5,
                 },
                 trt=False,
                 exclude_nms=False,
                 exclude_post_process=False):
        super(RTMDetHead, self).__init__()
        assert len(in_channels) > 0, "len(in_channels) should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.pred_kernel_size = pred_kernel_size
        self.stacked_convs = stacked_convs
        self.feat_channels = int(feat_channels * width_mult)
        self.share_conv = share_conv
        self.exp_on_reg = exp_on_reg

        self.iou_loss = GIoULoss()
        self.loss_weight = loss_weight
        self.use_varifocal_loss = use_varifocal_loss

        self.static_assigner = static_assigner
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
                cls_convs.append(
                    ConvBNLayer(
                        chn, self.feat_channels, 3, padding=1))
                reg_convs.append(
                    ConvBNLayer(
                        chn, self.feat_channels, 3, padding=1))
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

    def forward(self, feats, targets=None):
        assert len(feats) == len(self.fpn_strides), \
            "The size of feats is not equal to size of fpn_strides"

        cls_scores = []
        bbox_preds = []
        for idx, (x, stride) in enumerate(zip(feats, self.fpn_strides)):
            cls_feat = x
            reg_feat = x
            for cls_layer in self.cls_convs[idx]:
                cls_feat = cls_layer(cls_feat)
            cls_score = self.cls_preds[idx](cls_feat)

            for reg_layer in self.reg_convs[idx]:
                reg_feat = reg_layer(reg_feat)
            if self.exp_on_reg:
                reg_dist = self.reg_preds[idx](reg_feat).exp() * stride
            else:
                reg_dist = self.reg_preds[idx](reg_feat) * stride

            cls_scores.append(cls_score.flatten(2).transpose([0, 2, 1]))
            bbox_preds.append(reg_dist.flatten(2).transpose([0, 2, 1]))

        cls_scores = paddle.concat(cls_scores, 1)
        bbox_preds = paddle.concat(bbox_preds, 1)

        if self.training:
            return self.get_loss([cls_scores, bbox_preds], targets)
        else:
            return cls_scores, bbox_preds

    def get_loss(self, head_outs, gt_meta):
        cls_scores, bbox_preds = head_outs
        ### TODO

        out_dict = {
            'loss': loss,
            'loss_cls': loss_cls,
            'loss_iou': loss_iou,
        }
        return out_dict

    def post_process(self, head_outs, im_shape, scale_factor):
        pred_scores, pred_bboxes = head_outs
        ### TODO

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
