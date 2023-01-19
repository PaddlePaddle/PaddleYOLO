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
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from ppdet.core.workspace import register, serializable
from .csp_darknet import BaseConv, DWConv, BottleNeck, SPPFLayer
from ..shape_spec import ShapeSpec

__all__ = ['C2fLayer', 'C2Layer', 'YOLOv8CSPDarkNet']


class C2fLayer(nn.Layer):
    """C2f layer with 2 convs, named C2f in YOLOv8"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=1,
                 shortcut=False,
                 expansion=0.5,
                 depthwise=False,
                 bias=False,
                 act="silu"):
        super(C2fLayer, self).__init__()
        self.c = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(
            in_channels, 2 * self.c, ksize=1, stride=1, bias=bias, act=act)
        self.conv2 = BaseConv(
            (2 + num_blocks) * self.c,
            out_channels,
            ksize=1,
            stride=1,
            bias=bias,
            act=act)
        self.bottlenecks = nn.LayerList([
            BottleNeck(
                self.c,
                self.c,
                shortcut=shortcut,
                kernel_sizes=(3, 3),
                expansion=1.0,
                depthwise=depthwise,
                bias=bias,
                act=act) for _ in range(num_blocks)
        ])

    def forward(self, x):
        y = list(self.conv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.bottlenecks)
        return self.conv2(paddle.concat(y, 1))


class C2Layer(nn.Layer):
    """C2 layer with 2 convs, named C2 in YOLOv8"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=1,
                 shortcut=False,
                 expansion=0.5,
                 depthwise=False,
                 bias=False,
                 act="silu"):
        super(C2Layer, self).__init__()
        self.c = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(
            in_channels, 2 * self.c, ksize=1, stride=1, bias=bias, act=act)
        self.conv2 = BaseConv(
            2 * self.c, out_channels, ksize=1, stride=1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*(BottleNeck(
            self.c,
            self.c,
            shortcut=shortcut,
            kernel_sizes=(3, 3),
            expansion=1.0,
            depthwise=depthwise,
            bias=bias,
            act=act) for _ in range(num_blocks)))

    def forward(self, x):
        a, b = self.conv1(x).split((self.c, self.c), 1)
        return self.conv2(paddle.concat((self.bottlenecks(a), b), 1))


@register
@serializable
class YOLOv8CSPDarkNet(nn.Layer):
    """
    YOLOv8 CSPDarkNet backbone.
    diff with YOLOv5 CSPDarkNet:
    1. self.stem ksize 3 in YOLOv8 while 6 in YOLOv5
    2. use C2fLayer in YOLOv8 while CSPLayer in YOLOv5
    3. num_blocks [3,6,6,3] in YOLOv8 while [3,6,9,3] in YOLOv5
    4. channels of last stage in M/L/X

    Args:
        arch (str): Architecture of YOLOv8 CSPDarkNet, from {P5, P6}
        depth_mult (float): Depth multiplier, multiply number of channels in
            each layer, default as 1.0.
        width_mult (float): Width multiplier, multiply number of blocks in
            C2fLayer, default as 1.0.
        depthwise (bool): Whether to use depth-wise conv layer.
        act (str): Activation function type, default as 'silu'.
        return_idx (list): Index of stages whose feature maps are returned.
    """

    __shared__ = ['depth_mult', 'width_mult', 'act', 'trt']

    # in_channels, out_channels, num_blocks, add_shortcut, use_sppf
    arch_settings = {
        'P5': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 6, True, False], [512, 1024, 3, True, True]],
        'P6': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 6, True, False], [512, 768, 3, True, False],
               [768, 1024, 3, True, True]],
    }

    def __init__(self,
                 arch='P5',
                 depth_mult=1.0,
                 width_mult=1.0,
                 last_stage_ch=1024,
                 last2_stage_ch=512,
                 depthwise=False,
                 act='silu',
                 trt=False,
                 return_idx=[2, 3, 4]):
        super(YOLOv8CSPDarkNet, self).__init__()
        self.return_idx = return_idx
        Conv = DWConv if depthwise else BaseConv

        arch_setting = self.arch_settings[arch]
        # channels of last stage in M/L/X will be smaller
        if last_stage_ch != 1024:
            assert last_stage_ch > 0
            arch_setting[-1][1] = last_stage_ch
            if arch == 'P6' and last2_stage_ch != 768:
                assert last2_stage_ch > 0
                arch_setting[-2][1] = last2_stage_ch
                arch_setting[-1][0] = last2_stage_ch
        base_channels = int(arch_setting[0][0] * width_mult)

        self.stem = Conv(
            3, base_channels, ksize=3, stride=2, bias=False, act=act)

        _out_channels = [base_channels]
        layers_num = 1
        self.csp_dark_blocks = []

        for i, (in_channels, out_channels, num_blocks, shortcut,
                use_sppf) in enumerate(arch_setting):
            in_channels = int(in_channels * width_mult)
            out_channels = int(out_channels * width_mult)
            _out_channels.append(out_channels)
            num_blocks = max(round(num_blocks * depth_mult), 1)
            stage = []

            conv_layer = self.add_sublayer(
                'layers{}.stage{}.conv_layer'.format(layers_num, i + 1),
                Conv(
                    in_channels, out_channels, 3, 2, bias=False, act=act))
            stage.append(conv_layer)
            layers_num += 1

            c2f_layer = self.add_sublayer(
                'layers{}.stage{}.c2f_layer'.format(layers_num, i + 1),
                C2fLayer(
                    out_channels,
                    out_channels,
                    num_blocks=num_blocks,
                    shortcut=shortcut,
                    depthwise=depthwise,
                    bias=False,
                    act=act))
            stage.append(c2f_layer)
            layers_num += 1

            if use_sppf:
                sppf_layer = self.add_sublayer(
                    'layers{}.stage{}.sppf_layer'.format(layers_num, i + 1),
                    SPPFLayer(
                        out_channels,
                        out_channels,
                        ksize=5,
                        bias=False,
                        act=act))
                stage.append(sppf_layer)
                layers_num += 1

            self.csp_dark_blocks.append(nn.Sequential(*stage))

        self._out_channels = [_out_channels[i] for i in self.return_idx]
        self.strides = [[2, 4, 8, 16, 32, 64][i] for i in self.return_idx]

    def forward(self, inputs):
        x = inputs['image']
        outputs = []
        x = self.stem(x)
        for i, layer in enumerate(self.csp_dark_blocks):
            x = layer(x)
            if i + 1 in self.return_idx:
                outputs.append(x)
        return outputs

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=c, stride=s)
            for c, s in zip(self._out_channels, self.strides)
        ]
