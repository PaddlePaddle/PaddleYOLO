# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved. 
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
from ..backbones.cspnext import CSPNeXtLayer
from ..backbones.csp_darknet import BaseConv, DWConv
from ..shape_spec import ShapeSpec

__all__ = ['CSPNeXtPAFPN']


@register
@serializable
class CSPNeXtPAFPN(nn.Layer):
    """
    CSPNeXtPAFPN of RTMDet.
    """
    __shared__ = ['depth_mult', 'width_mult', 'data_format', 'act', 'trt']

    def __init__(self,
                 depth_mult=1.0,
                 width_mult=1.0,
                 in_channels=[256, 512, 1024],
                 out_channels=256,
                 depthwise=False,
                 data_format='NCHW',
                 act='silu',
                 trt=False):
        super(CSPNeXtPAFPN, self).__init__()
        self.in_channels = in_channels
        self._out_channels = [
            int(out_channels * width_mult) for _ in range(len(in_channels))
        ]
        Conv = DWConv if depthwise else BaseConv

        self.data_format = data_format
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # top-down fpn
        self.lateral_convs = nn.LayerList()
        self.fpn_blocks = nn.LayerList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(
                BaseConv(
                    int(in_channels[idx]),
                    int(in_channels[idx - 1]),
                    1,
                    1,
                    act=act))
            self.fpn_blocks.append(
                CSPNeXtLayer(
                    int(in_channels[idx - 1] * 2),
                    int(in_channels[idx - 1]),
                    round(3 * depth_mult),
                    shortcut=False,
                    depthwise=depthwise,
                    act=act))

        # bottom-up pan
        self.downsample_convs = nn.LayerList()
        self.pan_blocks = nn.LayerList()
        for idx in range(len(in_channels) - 1):
            self.downsample_convs.append(
                Conv(
                    int(in_channels[idx]),
                    int(in_channels[idx]),
                    3,
                    stride=2,
                    act=act))
            self.pan_blocks.append(
                CSPNeXtLayer(
                    int(in_channels[idx] * 2),
                    int(in_channels[idx + 1]),
                    round(3 * depth_mult),
                    shortcut=False,
                    depthwise=depthwise,
                    act=act))

        # CSPNeXtPAFPN new added
        self.out_convs = nn.LayerList()
        for in_ch, out_ch in zip(self.in_channels, self._out_channels):
            self.out_convs.append(Conv(in_ch, out_ch, 3, 1, act=act))

    def forward(self, feats, for_mot=False):
        assert len(feats) == len(self.in_channels)

        # top-down fpn
        inner_outs = [feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](
                feat_heigh)
            inner_outs[0] = feat_heigh

            upsample_feat = F.interpolate(
                feat_heigh,
                scale_factor=2.,
                mode="nearest",
                data_format=self.data_format)
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
                paddle.concat(
                    [upsample_feat, feat_low], axis=1))
            inner_outs.insert(0, inner_out)

        # bottom-up pan
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](paddle.concat(
                [downsample_feat, feat_height], axis=1))
            outs.append(out)
        # [4, 96, 80, 80] [4, 192, 40, 40] [4, 384, 20, 20]

        # out convs
        for i, conv in enumerate(self.out_convs):
            outs[i] = conv(outs[i])

        return outs

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    @property
    def out_shape(self):
        return [ShapeSpec(channels=c) for c in self._out_channels]
