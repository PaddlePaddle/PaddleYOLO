# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved. 
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
from ..backbones.yolo11_csp_darknet import C3k2

__all__ = ['YOLO11CSPPAN']


@register
@serializable
class YOLO11CSPPAN(nn.Layer):
    """
    YOLO11 CSP-PAN FPN, used in YOLO11
    diff with YOLOv8 CSP-PAN FPN:
    1. use C3k2 in YOLO11 while C2fLayer in YOLOv8
    """
    __shared__ = ['depth_mult', 'act', 'trt']

    def __init__(self,
                 depth_mult=1.0,
                 in_channels=[256, 512, 1024],
                 depthwise=False,
                 force_use_c3k=False,
                 act='silu',
                 trt=False):
        super(YOLO11CSPPAN, self).__init__()
        self.in_channels = in_channels
        in_channels = [in_channels[0] // 2, *in_channels[1:]]
        self._out_channels = in_channels

        # top-down
        self.fpn_p4 = C3k2(
            int(in_channels[2] + in_channels[1]),
            int(in_channels[1]),
            round(2 * depth_mult),
            shortcut=True,
            depthwise=depthwise,
            act=act,
            c3k=force_use_c3k)

        self.fpn_p3 = C3k2(
            int(in_channels[1] + self.in_channels[0]),
            int(in_channels[0]),
            round(2 * depth_mult),
            shortcut=True,
            depthwise=depthwise,
            act=act,
            c3k=force_use_c3k)

        # bottom-up
        self.down_conv2 = BaseConv(
            int(in_channels[0]), int(in_channels[0]), 3, stride=2, act=act)
        self.pan_n3 = C3k2(
            int(in_channels[0] + in_channels[1]),
            int(in_channels[1]),
            round(2 * depth_mult),
            shortcut=True,
            depthwise=depthwise,
            act=act,
            c3k=force_use_c3k)

        self.down_conv1 = BaseConv(
            int(in_channels[1]), int(in_channels[1]), 3, stride=2, act=act)
        self.pan_n4 = C3k2(
            int(in_channels[1] + in_channels[2]),
            int(in_channels[2]),
            round(2 * depth_mult),
            shortcut=True,
            depthwise=depthwise,
            act=act,
            c3k=True)

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
