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
from ppdet.core.workspace import register, serializable
from .csp_darknet import BaseConv, DWConv, SPPLayer
from ..shape_spec import ShapeSpec

__all__ = ['CSPNeXtBlock', 'CSPNeXtLayer', 'CSPNeXt']


class CSPNeXtBlock(nn.Layer):
    """The basic bottleneck block used in CSPNeXt."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 shortcut=True,
                 expansion=0.5,
                 depthwise=False,
                 kernel_size=5,
                 bias=False,
                 act="silu"):
        super(CSPNeXtBlock, self).__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(
            in_channels, hidden_channels, 3, stride=1, bias=bias, act=act)
        self.conv2 = DWConv(
            hidden_channels,
            out_channels,
            ksize=kernel_size,
            stride=1,
            bias=bias,
            act=act)
        self.add_shortcut = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.add_shortcut:
            y = y + x
        return y


class ChannelAttention(nn.Layer):
    def __init__(self, channels=256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2D(1)
        self.fc = nn.Conv2D(channels, channels, 1, 1, bias_attr=True)
        self.act = nn.Hardsigmoid()

    def forward(self, x):
        y = self.pool(x)
        out = self.act(self.fc(y))
        return x * out


class CSPNeXtLayer(nn.Layer):
    """CSPNeXt layer used in RTMDet, like CSPLayer(C3) in YOLOv5/YOLOX"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=1,
                 shortcut=True,
                 expansion=0.5,
                 depthwise=False,
                 ch_attn=False,
                 bias=False,
                 act="silu"):
        super(CSPNeXtLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.ch_attn = ch_attn
        self.conv1 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        self.conv2 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        self.conv3 = BaseConv(
            hidden_channels * 2,
            out_channels,
            ksize=1,
            stride=1,
            bias=bias,
            act=act)
        self.bottlenecks = nn.Sequential(* [
            CSPNeXtBlock(
                hidden_channels,
                hidden_channels,
                shortcut=shortcut,
                expansion=1.0,
                depthwise=depthwise,
                bias=bias,
                act=act) for _ in range(num_blocks)
        ])
        if ch_attn:
            self.ch_attn = ChannelAttention(hidden_channels * 2)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        x = paddle.concat([x_1, x_2], axis=1)
        if self.ch_attn:
            x = self.ch_attn(x)
        x = self.conv3(x)
        return x


@register
@serializable
class CSPNeXt(nn.Layer):
    """
    CSPNeXt backbone of RTMDet.
    Args:
        arch (str): Architecture of CSPNeXt, from {P5, P6}.
        depth_mult (float): Depth multiplier, multiply number of channels in
            each layer, default as 1.0.
        width_mult (float): Width multiplier, multiply number of blocks in
            CSPNeXtLayer, default as 1.0.
        depthwise (bool): Whether to use depth-wise conv layer.
        spp_kernel_sizes (tuple): kernel_sizes of SPP
        ch_attn (bool): Whether to add channel attention.
        act (str): Activation function type, default as 'silu'.
        trt (str): Whether to use trt infer in activation.
        return_idx (list): Index of stages whose feature maps are returned.
    """

    __shared__ = ['depth_mult', 'width_mult', 'act', 'trt']

    # in_channels, out_channels, num_blocks, add_shortcut, use_spp(use_sppf)
    arch_settings = {
        'P5': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 6, True, False], [512, 1024, 3, False, True]],
        'P6': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 6, True, False], [512, 768, 3, True, False],
               [768, 1024, 3, False, True]]
    }

    def __init__(self,
                 arch='P5',
                 depth_mult=1.0,
                 width_mult=1.0,
                 depthwise=False,
                 spp_kernel_sizes=(5, 9, 13),
                 ch_attn=True,
                 act='silu',
                 trt=False,
                 return_idx=[2, 3, 4]):
        super(CSPNeXt, self).__init__()
        self.arch = arch
        self.return_idx = return_idx
        Conv = DWConv if depthwise else BaseConv
        arch_setting = self.arch_settings[arch]
        stem_ch = int(arch_setting[0][0] * width_mult // 2)
        stem_out_ch = int(stem_ch * 2)

        self.stem = nn.Sequential(
            ('conv1', BaseConv(
                3, stem_ch, 3, 2, act=act)), ('conv2', BaseConv(
                    stem_ch, stem_ch, 3, 1, act=act)), ('conv3', BaseConv(
                        stem_ch, stem_out_ch, 3, 1, act=act)))

        _out_channels = [stem_out_ch]
        layers_num = 1
        self.csp_next_blocks = []

        for i, (in_ch, out_ch, n, shortcut, use_spp) in enumerate(arch_setting):
            in_channels = int(in_ch * width_mult)
            out_channels = int(out_ch * width_mult)
            _out_channels.append(out_channels)
            num_blocks = max(round(n * depth_mult), 1)
            stage = []

            conv_layer = self.add_sublayer(
                'layers{}.stage{}.conv_layer'.format(layers_num, i + 1),
                Conv(
                    in_channels, out_channels, 3, 2, act=act))
            stage.append(conv_layer)
            layers_num += 1

            if use_spp:
                spp_layer = self.add_sublayer(
                    'layers{}.stage{}.spp_layer'.format(layers_num, i + 1),
                    SPPLayer(
                        out_channels,
                        out_channels,
                        kernel_sizes=spp_kernel_sizes,
                        bias=False,
                        act=act))
                stage.append(spp_layer)
                layers_num += 1

            csp_layer = self.add_sublayer(
                'layers{}.stage{}.cspnext_layer'.format(layers_num, i + 1),
                CSPNeXtLayer(
                    out_channels,
                    out_channels,
                    num_blocks=num_blocks,
                    shortcut=shortcut,
                    depthwise=depthwise,
                    ch_attn=ch_attn,
                    bias=False,
                    act=act))
            stage.append(csp_layer)
            layers_num += 1

            self.csp_next_blocks.append(nn.Sequential(*stage))

        self._out_channels = [_out_channels[i] for i in self.return_idx]
        self.strides = [[2, 4, 8, 16, 32, 64][i] for i in self.return_idx]

    def forward(self, inputs):
        x = inputs['image']
        outputs = []
        x = self.stem(x)
        for i, layer in enumerate(self.csp_next_blocks):
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
