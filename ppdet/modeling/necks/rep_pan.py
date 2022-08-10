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
"""
This code is based on https://github.com/meituan/YOLOv6
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from ppdet.core.workspace import register, serializable
from ..shape_spec import ShapeSpec
from ..backbones.efficientrep import make_divisible, SimConv, Transpose, RepLayer


@register
@serializable
class RepPAN(nn.Layer):
    """RepPAN of MT-YOLOv6
    """
    __shared__ = ['depth_mult', 'width_mult', 'act', 'trt']

    def __init__(self,
                 depth_mult=1.0,
                 width_mult=1.0,
                 in_channels=[256, 512, 1024],
                 out_channels=[256, 512, 1024],
                 num_repeats=[12, 12, 12, 12],
                 depthwise=False,
                 act='silu',
                 trt=False):
        super(RepPAN, self).__init__()
        backbone_ch_list = [64, 128, 256, 512, 1024]
        ch_list = backbone_ch_list + [256, 128, 128, 256, 256, 512]
        num_repeats = [(max(round(i * depth_mult), 1) if i > 1 else i)
                       for i in (num_repeats)]
        ch_list = [make_divisible(i * width_mult, 8) for i in (ch_list)]
        self.in_channels = in_channels
        self._out_channels = ch_list[6], ch_list[8], ch_list[10]

        in_ch, out_ch = self.in_channels[2], ch_list[5]
        self.lateral_conv1 = SimConv(in_ch, out_ch, 1, 1)
        self.up1 = Transpose(out_ch, out_ch)
        self.rep_fpn1 = RepLayer(
            self.in_channels[1] + out_ch,  # 512+256
            out_ch,  # 256
            num_repeats[0])

        in_ch, out_ch = ch_list[5], ch_list[6]
        self.lateral_conv2 = SimConv(in_ch, out_ch, 1, 1)
        self.up2 = Transpose(out_ch, out_ch)
        self.rep_fpn2 = RepLayer(
            self.in_channels[0] + out_ch,  # 128+64
            out_ch,  # 64
            num_repeats[1])

        in_ch, out_ch1, out_ch2 = ch_list[6], ch_list[7], ch_list[8]
        self.down_conv1 = SimConv(in_ch, out_ch1, 3, 2)
        self.rep_pan1 = RepLayer(
            in_ch + out_ch1,  # 64+64
            out_ch2,  # 128
            num_repeats[2])

        in_ch, out_ch1, out_ch2 = ch_list[8], ch_list[9], ch_list[10]
        self.down_conv2 = SimConv(in_ch, out_ch1, 3, 2)
        self.rep_pan2 = RepLayer(
            ch_list[5] + out_ch1,  # 128+128
            out_ch2,  # 256
            num_repeats[3])

    def forward(self, feats, for_mot=False):
        assert len(feats) == len(self.in_channels)
        [c3, c4, c5] = feats
        # [8, 128, 80, 80] [8, 256, 40, 40] [8, 512, 20, 20]

        # top-down FPN
        fpn_out1 = self.lateral_conv1(c5)
        up_feat1 = self.up1(fpn_out1)
        f_concat1 = paddle.concat([up_feat1, c4], 1)
        f_out1 = self.rep_fpn1(f_concat1)

        fpn_out2 = self.lateral_conv2(f_out1)
        up_feat2 = self.up2(fpn_out2)
        f_concat2 = paddle.concat([up_feat2, c3], 1)
        pan_out2 = self.rep_fpn2(f_concat2)

        # bottom-up PAN
        down_feat1 = self.down_conv1(pan_out2)
        p_concat1 = paddle.concat([down_feat1, fpn_out2], 1)
        pan_out1 = self.rep_pan1(p_concat1)

        down_feat2 = self.down_conv2(pan_out1)
        p_concat2 = paddle.concat([down_feat2, fpn_out1], 1)
        pan_out0 = self.rep_pan2(p_concat2)

        return [pan_out2, pan_out1, pan_out0]

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    @property
    def out_shape(self):
        return [ShapeSpec(channels=c) for c in self._out_channels]
