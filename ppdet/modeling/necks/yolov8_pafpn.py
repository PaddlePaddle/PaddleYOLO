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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register, serializable
from ..shape_spec import ShapeSpec
from ..backbones.csp_darknet import BaseConv
from ..backbones.yolov8_csp_darknet import C2fLayer, C2Layer

__all__ = ['YOLOv8CSPPAN', 'YOLOv8CSPPANP6']


@register
@serializable
class YOLOv8CSPPAN(nn.Layer):
    """
    YOLOv8 CSP-PAN FPN, used in YOLOv8
    diff with YOLOv5 CSP-PAN FPN:
    1. no lateral convs
    2. use C2fLayer in YOLOv8 while CSPLayer in YOLOv5
    """
    __shared__ = ['depth_mult', 'act', 'trt']

    def __init__(self,
                 depth_mult=1.0,
                 in_channels=[256, 512, 1024],
                 depthwise=False,
                 act='silu',
                 trt=False):
        super(YOLOv8CSPPAN, self).__init__()
        self.in_channels = in_channels
        self._out_channels = in_channels

        # top-down
        self.fpn_p4 = C2fLayer(
            int(in_channels[2] + in_channels[1]),
            int(in_channels[1]),
            round(3 * depth_mult),
            shortcut=False,
            depthwise=depthwise,
            act=act)

        self.fpn_p3 = C2fLayer(
            int(in_channels[1] + in_channels[0]),
            int(in_channels[0]),
            round(3 * depth_mult),
            shortcut=False,
            depthwise=depthwise,
            act=act)

        # bottom-up
        self.down_conv2 = BaseConv(
            int(in_channels[0]), int(in_channels[0]), 3, stride=2, act=act)
        self.pan_n3 = C2fLayer(
            int(in_channels[0] + in_channels[1]),
            int(in_channels[1]),
            round(3 * depth_mult),
            shortcut=False,
            depthwise=depthwise,
            act=act)

        self.down_conv1 = BaseConv(
            int(in_channels[1]), int(in_channels[1]), 3, stride=2, act=act)
        self.pan_n4 = C2fLayer(
            int(in_channels[1] + in_channels[2]),
            int(in_channels[2]),
            round(3 * depth_mult),
            shortcut=False,
            depthwise=depthwise,
            act=act)

    def forward(self, feats, for_mot=False):
        [c3, c4, c5] = feats

        # top-down FPN
        up_feat1 = F.interpolate(c5, scale_factor=2., mode="nearest")
        f_concat1 = paddle.concat([up_feat1, c4], 1)
        f_out1 = self.fpn_p4(f_concat1)

        up_feat2 = F.interpolate(f_out1, scale_factor=2., mode="nearest")
        f_concat2 = paddle.concat([up_feat2, c3], 1)
        f_out0 = self.fpn_p3(f_concat2)

        # bottom-up PAN
        down_feat1 = self.down_conv2(f_out0)
        p_concat1 = paddle.concat([down_feat1, f_out1], 1)
        pan_out1 = self.pan_n3(p_concat1)

        down_feat2 = self.down_conv1(pan_out1)
        p_concat2 = paddle.concat([down_feat2, c5], 1)
        pan_out0 = self.pan_n4(p_concat2)

        return [f_out0, pan_out1, pan_out0]

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    @property
    def out_shape(self):
        return [ShapeSpec(channels=c) for c in self._out_channels]


@register
@serializable
class YOLOv8CSPPANP6(nn.Layer):
    """
    YOLOv8 CSP-PAN FPN, used in YOLOv8-P6
    diff with YOLOv5 CSP-PAN FPN:
    1. no lateral convs
    2. use C2Layer in YOLOv8-P6 while CSPLayer in YOLOv5-P6
    """
    __shared__ = ['depth_mult', 'act', 'trt']

    def __init__(self,
                 depth_mult=1.0,
                 in_channels=[256, 512, 768, 1024],
                 depthwise=False,
                 act='silu',
                 trt=False):
        super(YOLOv8CSPPANP6, self).__init__()
        self.in_channels = in_channels
        self._out_channels = in_channels

        # top-down
        self.fpn_p5 = C2Layer(
            int(in_channels[3] + in_channels[2]),
            int(in_channels[2]),
            round(3 * depth_mult),
            shortcut=False,
            depthwise=depthwise,
            act=act)

        self.fpn_p4 = C2Layer(
            int(in_channels[2] + in_channels[1]),
            int(in_channels[1]),
            round(3 * depth_mult),
            shortcut=False,
            depthwise=depthwise,
            act=act)

        self.fpn_p3 = C2Layer(
            int(in_channels[1] + in_channels[0]),
            int(in_channels[0]),
            round(3 * depth_mult),
            shortcut=False,
            depthwise=depthwise,
            act=act)

        # bottom-up
        self.down_conv2 = BaseConv(
            int(in_channels[0]), int(in_channels[0]), 3, stride=2, act=act)
        self.pan_n3 = C2Layer(
            int(in_channels[0] + in_channels[1]),
            int(in_channels[1]),
            round(3 * depth_mult),
            shortcut=False,
            depthwise=depthwise,
            act=act)

        self.down_conv1 = BaseConv(
            int(in_channels[1]), int(in_channels[1]), 3, stride=2, act=act)
        self.pan_n4 = C2Layer(
            int(in_channels[1] + in_channels[2]),
            int(in_channels[2]),
            round(3 * depth_mult),
            shortcut=False,
            depthwise=depthwise,
            act=act)

        self.down_conv0 = BaseConv(
            int(in_channels[2]), int(in_channels[2]), 3, stride=2, act=act)
        self.pan_n5 = C2Layer(
            int(in_channels[2] + in_channels[3]),
            int(in_channels[3]),
            round(3 * depth_mult),
            shortcut=False,
            depthwise=depthwise,
            act=act)

    def forward(self, feats, for_mot=False):
        [c3, c4, c5, c6] = feats

        # top-down FPN
        up_feat0 = F.interpolate(c6, scale_factor=2., mode="nearest")
        f_concat0 = paddle.concat([up_feat0, c5], 1)
        f_out0 = self.fpn_p5(f_concat0)

        up_feat1 = F.interpolate(f_out0, scale_factor=2., mode="nearest")
        f_concat1 = paddle.concat([up_feat1, c4], 1)
        f_out1 = self.fpn_p4(f_concat1)

        up_feat2 = F.interpolate(f_out1, scale_factor=2., mode="nearest")
        f_concat2 = paddle.concat([up_feat2, c3], 1)
        f_out2 = self.fpn_p3(f_concat2)

        # bottom-up PAN
        down_feat1 = self.down_conv2(f_out2)
        p_concat1 = paddle.concat([down_feat1, f_out1], 1)
        pan_out2 = self.pan_n3(p_concat1)

        down_feat2 = self.down_conv1(pan_out2)
        p_concat2 = paddle.concat([down_feat2, c5], 1)
        pan_out1 = self.pan_n4(p_concat2)

        down_feat3 = self.down_conv0(pan_out1)
        p_concat3 = paddle.concat([down_feat3, c6], 1)
        pan_out0 = self.pan_n5(p_concat3)

        return [f_out2, pan_out2, pan_out1, pan_out0]

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    @property
    def out_shape(self):
        return [ShapeSpec(channels=c) for c in self._out_channels]
