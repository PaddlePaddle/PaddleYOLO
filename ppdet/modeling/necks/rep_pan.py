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
from ..backbones.efficientrep import RepBlock, SimConv, Transpose, make_divisible


@register
@serializable
class RepPANNeck(nn.Layer):
    """RepPANNeck Module
    EfficientRep is the default backbone of this model.
    RepPANNeck has the balance of feature fusion ability and hardware efficiency.
    """
    __shared__ = ['depth_mult', 'width_mult', 'act', 'trt']

    def __init__(self,
                 depth_mult=1.0,
                 width_mult=1.0,
                 in_channels=[256, 512, 1024],
                 out_channels=[1024, 512, 256],
                 num_repeats=[12, 12, 12, 12],
                 depthwise=False,
                 data_format='NCHW',
                 act='silu',
                 trt=False):
        super().__init__()
        channels_list = [64, 128, 256, 512, 1024] + [
            256, 128, 128, 256, 256, 512
        ]
        num_repeats = [(max(round(i * depth_mult), 1) if i > 1 else i)
                       for i in (num_repeats)]
        channels_list = [
            make_divisible(i * width_mult, 8) for i in (channels_list)
        ]
        self.in_channels = in_channels
        self._out_channels = in_channels

        # top-down FPN
        self.reduce_layer0 = SimConv(
            channels_list[4], channels_list[5], kernel_size=1, stride=1)
        self.upsample0 = Transpose(channels_list[5], channels_list[5])
        self.Rep_p4 = RepBlock(
            channels_list[3] + channels_list[5],  # 256+128
            out_channels=channels_list[5],  # 128
            n=num_repeats[0], )

        self.reduce_layer1 = SimConv(
            channels_list[5], channels_list[6], kernel_size=1, stride=1)
        self.upsample1 = Transpose(channels_list[6], channels_list[6])
        self.Rep_p3 = RepBlock(
            in_channels=channels_list[2] + channels_list[6],  # 128+64
            out_channels=channels_list[6],  # 64
            n=num_repeats[1])

        # bottom-up PAN
        self.downsample2 = SimConv(
            channels_list[6], channels_list[7], kernel_size=3, stride=2)
        self.Rep_n3 = RepBlock(
            in_channels=channels_list[6] + channels_list[7],  # 64+64
            out_channels=channels_list[8],  # 128
            n=num_repeats[2], )

        self.downsample1 = SimConv(
            channels_list[8], channels_list[9], kernel_size=3, stride=2)
        self.Rep_n4 = RepBlock(
            in_channels=channels_list[5] + channels_list[9],  # 128+128
            out_channels=channels_list[10],  # 256
            n=num_repeats[3])

    def forward(self, feats, for_mot=False):
        assert len(feats) == len(self.in_channels)
        [x2, x1, x0] = feats
        # [8, 128, 80, 80] [8, 256, 40, 40] [8, 512, 20, 20]

        # top-down FPN
        fpn_out0 = self.reduce_layer0(x0)
        upsample_feat0 = self.upsample0(fpn_out0)
        f_concat_layer0 = paddle.concat([upsample_feat0, x1], 1)
        f_out0 = self.Rep_p4(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(f_out0)
        upsample_feat1 = self.upsample1(fpn_out1)
        f_concat_layer1 = paddle.concat([upsample_feat1, x2], 1)
        pan_out2 = self.Rep_p3(f_concat_layer1)

        # bottom-up PAN
        down_feat1 = self.downsample2(pan_out2)
        p_concat_layer1 = paddle.concat([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n3(p_concat_layer1)

        down_feat0 = self.downsample1(pan_out1)
        p_concat_layer2 = paddle.concat([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n4(p_concat_layer2)

        return [pan_out2, pan_out1, pan_out0]

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    @property
    def out_shape(self):
        return [ShapeSpec(channels=c) for c in self._out_channels]
