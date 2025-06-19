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
from ppdet.core.workspace import register, serializable
from ..shape_spec import ShapeSpec
from .csp_darknet import BaseConv, BottleNeck, CSPLayer, DWConv, SPPFLayer
from .yolov8_csp_darknet import C2fLayer
from .yolov10_csp_darknet import AttnLayer


__all__ = ['C3k2', 'YOLO11CSPDarkNet']


class C3k(CSPLayer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=1,
                 shortcut=True,
                 expansion=0.5,
                 depthwise=False,
                 kernel_sizes=3,
                 bias=False,
                 act="silu"):
        super(C3k, self).__init__(
            in_channels, out_channels, num_blocks, shortcut, expansion, depthwise, bias, act)
        hidden_channels = int(out_channels * expansion)
        self.bottlenecks = nn.Sequential(* [
            BottleNeck(
                hidden_channels,
                hidden_channels,
                shortcut=shortcut,
                kernel_sizes=(kernel_sizes, kernel_sizes),
                expansion=1.0,
                depthwise=depthwise,
                bias=bias,
                act=act) for _ in range(num_blocks)
        ])


class C3k2(C2fLayer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=1,
                 shortcut=True,
                 expansion=0.5,
                 depthwise=False,
                 bias=False,
                 act="silu",
                 c3k=False):
        super(C3k2, self).__init__(
            in_channels,
            out_channels,
            num_blocks=num_blocks,
            shortcut=shortcut,
            expansion=expansion,
            depthwise=depthwise,
            bias=bias,
            act=act)
        self.bottlenecks = nn.LayerList([
            C3k(self.c, self.c, num_blocks=2, shortcut=shortcut, depthwise=depthwise)
            if c3k else BottleNeck(
                self.c,
                self.c,
                shortcut=shortcut,
                kernel_sizes=(3, 3),
                expansion=0.5,
                depthwise=depthwise,
                bias=bias,
                act=act) for _ in range(num_blocks)])


class PSABlock(nn.Layer):
    def __init__(self, embed_dim, num_heads, expansion=1.0, act='silu'):
        super(PSABlock, self).__init__()
        hidden_dim = int(embed_dim * expansion)
        self.attn = AttnLayer(hidden_dim,
                              num_heads=num_heads,
                              attn_ratio=0.5)
        self.ffn = nn.Sequential(*[
            BaseConv(hidden_dim,
                     2 * hidden_dim,
                     ksize=1,
                     stride=1,
                     groups=1,
                     bias=False,
                     act=act),
            BaseConv(2 * hidden_dim,
                     hidden_dim,
                     ksize=1,
                     stride=1,
                     groups=1,
                     bias=False,
                     act=nn.Identity())
        ])

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x


class C2PSA(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=1,
                 expansion=0.5):
        assert in_channels == out_channels, 'Input and output channels must be equal for C2PSA'
        super(C2PSA, self).__init__()
        self.c = int(in_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, 2 * self.c, 1, 1)
        self.conv2 = BaseConv(2 * self.c, in_channels, 1, 1)
        self.bottlenecks = nn.Sequential(*[
            PSABlock(self.c, self.c // 64) for _ in range(num_blocks)])

    def forward(self, x):
        a, b = self.conv1(x).split((self.c, self.c), 1)
        return self.conv2(paddle.concat((a, self.bottlenecks(b)), 1))


@register
@serializable
class YOLO11CSPDarkNet(nn.Layer):
    """
    YOLO11 CSPDarkNet backbone.
    """
    __shared__ = ['depth_mult', 'width_mult', 'act', 'trt', 'use_checkpoint']

    # in_channels, out_channels, num_blocks, add_shortcut, use_c3k, expansion, use_sppf, use_c2psa
    arch_settings = {
        'P5': [[64, 256, 2, True, False, 0.25, False, False],
               [256, 512, 2, True, False, 0.25, False, False],
               [512, 512, 2, True, True, 0.5, False, False],
               [512, 1024, 2, True, True, 0.5, True, True]],
    }

    def __init__(self,
                 arch='P5',
                 depth_mult=1.0,
                 width_mult=1.0,
                 last_stage_ch=1024,
                 depthwise=False,
                 force_use_c3k=False,
                 act='silu',
                 trt=False,
                 return_idx=[2, 3, 4],
                 use_checkpoint=False):
        super(YOLO11CSPDarkNet, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.return_idx = return_idx
        Conv = DWConv if depthwise else BaseConv

        arch_setting = self.arch_settings[arch]
        # channels of last stage in M/L/X will be smaller
        if last_stage_ch != 1024:
            assert last_stage_ch > 0
            arch_setting[-1][1] = last_stage_ch
        base_channels = int(arch_setting[0][0] * width_mult)

        self.stem = Conv(
            3, base_channels, ksize=3, stride=2, bias=False, act=act)

        _out_channels = [base_channels]
        layers_num = 1
        self.csp_dark_blocks = []

        for i, (in_channels, out_channels, num_blocks, shortcut,
                use_c3k, expansion, use_sppf, use_c2psa) in enumerate(arch_setting):
            in_channels = int(in_channels * width_mult)
            out_channels = int(out_channels * width_mult)
            mid_channels = out_channels // 2 if i < 2 else out_channels
            _out_channels.append(out_channels)
            num_blocks = max(round(num_blocks * depth_mult), 1)
            stage = []

            conv_layer = self.add_sublayer(
                'layers{}.stage{}.conv_layer'.format(layers_num, i + 1),
                Conv(
                    in_channels, mid_channels, 3, 2, bias=False, act=act))
            stage.append(conv_layer)
            layers_num += 1

            c2f_layer = self.add_sublayer(
                'layers{}.stage{}.c3k2_layer'.format(layers_num, i + 1),
                C3k2(
                    mid_channels,
                    out_channels,
                    num_blocks=num_blocks,
                    shortcut=shortcut,
                    expansion=expansion,
                    depthwise=depthwise,
                    bias=False,
                    act=act,
                    c3k=force_use_c3k or use_c3k))
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

            if use_c2psa:
                c2psa_layer = self.add_sublayer(
                    'layers{}.stage{}.c2psa_layer'.format(layers_num, i + 1),
                    C2PSA(
                        out_channels,
                        out_channels,
                        num_blocks=num_blocks))
                stage.append(c2psa_layer)
                layers_num += 1

            self.csp_dark_blocks.append(nn.Sequential(*stage))

        self._out_channels = [_out_channels[i] for i in self.return_idx]
        self.strides = [[2, 4, 8, 16, 32, 64][i] for i in self.return_idx]

    def forward(self, inputs):
        x = inputs['image']
        outputs = []
        x = self.stem(x)
        for i, layer in enumerate(self.csp_dark_blocks):
            if self.use_checkpoint and self.training:
                x = paddle.distributed.fleet.utils.recompute(
                    layer, x, **{"preserve_rng_state": True})
            else:
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
