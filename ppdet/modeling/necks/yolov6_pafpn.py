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
from ppdet.core.workspace import register, serializable
from ..backbones.yolov6_efficientrep import SimConv, Transpose, RepLayer, BepC3Layer, make_divisible, get_block
from ..shape_spec import ShapeSpec

__all__ = ['RepPAN', 'RepBiFPAN', 'CSPRepPAN', 'CSPRepBiFPAN']


class BiFusion(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cv1 = SimConv(in_channels[0], out_channels, 1, 1)
        self.cv2 = SimConv(in_channels[1], out_channels, 1, 1)
        self.cv3 = SimConv(out_channels * 3, out_channels, 1, 1)

        self.upsample = Transpose(
            in_channels=out_channels, out_channels=out_channels)
        self.downsample = SimConv(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2)

    def forward(self, x):
        x0 = self.upsample(x[0])
        x1 = self.cv1(x[1])
        x2 = self.downsample(self.cv2(x[2]))
        return self.cv3(paddle.concat([x0, x1, x2], 1))


@register
@serializable
class RepPAN(nn.Layer):
    """RepPAN of YOLOv6 n/t/s
    """
    __shared__ = ['depth_mult', 'width_mult', 'training_mode']

    def __init__(self,
                 depth_mult=1.0,
                 width_mult=1.0,
                 in_channels=[256, 512, 1024],
                 out_channels=[128, 256, 512],
                 num_repeats=[12, 12, 12, 12],
                 training_mode='repvgg'):
        super(RepPAN, self).__init__()
        backbone_ch_list = [64, 128, 256, 512, 1024]
        ch_list = backbone_ch_list + [256, 128, 128, 256, 256, 512]
        num_repeats = [(max(round(i * depth_mult), 1) if i > 1 else i)
                       for i in (num_repeats)]
        ch_list = [make_divisible(i * width_mult, 8) for i in (ch_list)]
        self.in_channels = in_channels
        self._out_channels = ch_list[6], ch_list[8], ch_list[10]

        # block = get_block(training_mode) # RepConv(RepVGGBlock) as default
        # Rep_p4
        in_ch, out_ch = self.in_channels[2], ch_list[5]
        self.lateral_conv1 = SimConv(in_ch, out_ch, 1, 1)
        self.up1 = Transpose(out_ch, out_ch)
        self.rep_fpn1 = RepLayer(self.in_channels[1] + out_ch, out_ch,
                                 num_repeats[0])

        # Rep_p3
        in_ch, out_ch = ch_list[5], ch_list[6]
        self.lateral_conv2 = SimConv(in_ch, out_ch, 1, 1)
        self.up2 = Transpose(out_ch, out_ch)
        self.rep_fpn2 = RepLayer(self.in_channels[0] + out_ch, out_ch,
                                 num_repeats[1])

        # Rep_n3
        in_ch, out_ch1, out_ch2 = ch_list[6], ch_list[7], ch_list[8]
        self.down_conv1 = SimConv(in_ch, out_ch1, 3, 2)
        self.rep_pan1 = RepLayer(in_ch + out_ch1, out_ch2, num_repeats[2])

        # Rep_n4
        in_ch, out_ch1, out_ch2 = ch_list[8], ch_list[9], ch_list[10]
        self.down_conv2 = SimConv(in_ch, out_ch1, 3, 2)
        self.rep_pan2 = RepLayer(ch_list[5] + out_ch1, out_ch2, num_repeats[3])

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


@register
@serializable
class RepBiFPAN(nn.Layer):
    """
    RepBiFPAN Neck for YOLOv6 n/s in v3.0
    change lateral_conv + up(Transpose) to BiFusion
    """
    __shared__ = ['depth_mult', 'width_mult', 'training_mode']

    def __init__(self,
                 depth_mult=0.33,
                 width_mult=0.50,
                 in_channels=[128, 256, 512, 1024],
                 out_channels=[128, 256, 512],
                 training_mode='repvgg'):
        super(RepBiFPAN, self).__init__()
        backbone_ch_list = [64, 128, 256, 512, 1024]
        backbone_num_repeats = [1, 6, 12, 18, 6]

        ch_list = backbone_ch_list + [256, 128, 128, 256, 256, 512]
        ch_list = [make_divisible(i * width_mult, 8) for i in (ch_list)]

        num_repeats = backbone_num_repeats + [12, 12, 12, 12]
        num_repeats = [(max(round(i * depth_mult), 1) if i > 1 else i)
                       for i in (num_repeats)]

        self.in_channels = in_channels
        self._out_channels = ch_list[6], ch_list[8], ch_list[10]

        # block = get_block(training_mode) # RepConv(RepVGGBlock) as default
        # Rep_p4
        self.reduce_layer0 = SimConv(ch_list[4], ch_list[5], 1, 1)
        self.Bifusion0 = BiFusion([ch_list[3], ch_list[5]], ch_list[5])
        self.Rep_p4 = RepLayer(ch_list[5], ch_list[5], num_repeats[5])

        # Rep_p3
        self.reduce_layer1 = SimConv(ch_list[5], ch_list[6], 1, 1)
        self.Bifusion1 = BiFusion([ch_list[5], ch_list[6]], ch_list[6])
        self.Rep_p3 = RepLayer(ch_list[6], ch_list[6], num_repeats[6])

        # Rep_n3
        self.downsample2 = SimConv(ch_list[6], ch_list[7], 3, 2)
        self.Rep_n3 = RepLayer(ch_list[6] + ch_list[7], ch_list[8],
                               num_repeats[7])

        # Rep_n4
        self.downsample1 = SimConv(ch_list[8], ch_list[9], 3, 2)
        self.Rep_n4 = RepLayer(ch_list[5] + ch_list[9], ch_list[10],
                               num_repeats[8])

    def forward(self, feats, for_mot=False):
        assert len(feats) == len(self.in_channels)
        [x3, x2, x1, x0] = feats  # p2, p3, p4, p5 

        # top-down
        fpn_out0 = self.reduce_layer0(x0)
        f_concat_layer0 = self.Bifusion0([fpn_out0, x1, x2])
        f_out0 = self.Rep_p4(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(f_out0)
        f_concat_layer1 = self.Bifusion1([fpn_out1, x2, x3])
        pan_out2 = self.Rep_p3(f_concat_layer1)

        # bottom-up
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


@register
@serializable
class CSPRepPAN(nn.Layer):
    """
    CSPRepPAN of YOLOv6 m/l
    """

    __shared__ = ['depth_mult', 'width_mult', 'act', 'training_mode']

    def __init__(self,
                 depth_mult=1.0,
                 width_mult=1.0,
                 in_channels=[256, 512, 1024],
                 out_channels=[128, 256, 512],
                 num_repeats=[12, 12, 12, 12],
                 training_mode='repvgg',
                 csp_e=0.5,
                 act='relu'):
        super(CSPRepPAN, self).__init__()
        backbone_ch_list = [64, 128, 256, 512, 1024]
        ch_list = backbone_ch_list + [256, 128, 128, 256, 256, 512]
        num_repeats = [(max(round(i * depth_mult), 1) if i > 1 else i)
                       for i in (num_repeats)]
        ch_list = [make_divisible(i * width_mult, 8) for i in (ch_list)]
        self.in_channels = in_channels
        self._out_channels = ch_list[6], ch_list[8], ch_list[10]
        if csp_e == 0.67:
            csp_e = float(2) / 3
        block = get_block(training_mode)
        # RepConv(or RepVGGBlock) in M, but ConvBNSiLUBlock(or ConvWrapper) in L

        # Rep_p4
        in_ch, out_ch = self.in_channels[2], ch_list[5]
        self.lateral_conv1 = SimConv(in_ch, out_ch, 1, 1)
        self.up1 = Transpose(out_ch, out_ch)
        self.Rep_p4 = BepC3Layer(
            self.in_channels[1] + out_ch,
            out_ch,
            num_repeats[0],
            csp_e,
            block=block,
            act=act)

        # Rep_p3
        in_ch, out_ch = ch_list[5], ch_list[6]
        self.lateral_conv2 = SimConv(in_ch, out_ch, 1, 1)
        self.up2 = Transpose(out_ch, out_ch)
        self.Rep_p3 = BepC3Layer(
            self.in_channels[0] + out_ch,
            out_ch,
            num_repeats[1],
            csp_e,
            block=block,
            act=act)

        # Rep_n3
        in_ch, out_ch1, out_ch2 = ch_list[6], ch_list[7], ch_list[8]
        self.down_conv1 = SimConv(in_ch, out_ch1, 3, 2)
        self.Rep_n3 = BepC3Layer(
            in_ch + out_ch1,
            out_ch2,
            num_repeats[2],
            csp_e,
            block=block,
            act=act)

        # Rep_n4
        in_ch, out_ch1, out_ch2 = ch_list[8], ch_list[9], ch_list[10]
        self.down_conv2 = SimConv(in_ch, out_ch1, 3, 2)
        self.Rep_n4 = BepC3Layer(
            ch_list[5] + out_ch1,
            out_ch2,
            num_repeats[3],
            csp_e,
            block=block,
            act=act)

    def forward(self, feats, for_mot=False):
        assert len(feats) == len(self.in_channels)
        [c3, c4, c5] = feats
        # [8, 128, 80, 80] [8, 256, 40, 40] [8, 512, 20, 20]

        # top-down FPN
        fpn_out1 = self.lateral_conv1(c5)  # reduce_layer0
        up_feat1 = self.up1(fpn_out1)
        f_concat1 = paddle.concat([up_feat1, c4], 1)
        f_out1 = self.Rep_p4(f_concat1)

        fpn_out2 = self.lateral_conv2(f_out1)  # reduce_layer1
        up_feat2 = self.up2(fpn_out2)
        f_concat2 = paddle.concat([up_feat2, c3], 1)
        pan_out2 = self.Rep_p3(f_concat2)

        # bottom-up PAN
        down_feat1 = self.down_conv1(pan_out2)  # downsample2
        p_concat1 = paddle.concat([down_feat1, fpn_out2], 1)
        pan_out1 = self.Rep_n3(p_concat1)

        down_feat2 = self.down_conv2(pan_out1)  # downsample1
        p_concat2 = paddle.concat([down_feat2, fpn_out1], 1)
        pan_out0 = self.Rep_n4(p_concat2)

        return [pan_out2, pan_out1, pan_out0]

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    @property
    def out_shape(self):
        return [ShapeSpec(channels=c) for c in self._out_channels]


@register
@serializable
class CSPRepBiFPAN(nn.Layer):
    """
    CSPRepBiFPAN of YOLOv6 m/l in v3.0
    change lateral_conv + up(Transpose) to BiFusion
    """
    __shared__ = ['depth_mult', 'width_mult', 'act', 'training_mode']

    def __init__(self,
                 depth_mult=1.0,
                 width_mult=1.0,
                 in_channels=[128, 256, 512, 1024],
                 out_channels=[128, 256, 512],
                 training_mode='repvgg',
                 csp_e=0.5,
                 act='relu'):
        super(CSPRepBiFPAN, self).__init__()
        backbone_ch_list = [64, 128, 256, 512, 1024]
        backbone_num_repeats = [1, 6, 12, 18, 6]

        ch_list = backbone_ch_list + [256, 128, 128, 256, 256, 512]
        ch_list = [make_divisible(i * width_mult, 8) for i in (ch_list)]

        num_repeats = backbone_num_repeats + [12, 12, 12, 12]
        num_repeats = [(max(round(i * depth_mult), 1) if i > 1 else i)
                       for i in (num_repeats)]

        self.in_channels = in_channels
        self._out_channels = ch_list[6], ch_list[8], ch_list[10]
        if csp_e == 0.67:
            csp_e = float(2) / 3
        block = get_block(training_mode)
        # RepConv(or RepVGGBlock) in M, but ConvBNSiLUBlock(or ConvWrapper) in L

        # Rep_p4
        self.reduce_layer0 = SimConv(ch_list[4], ch_list[5], 1, 1)
        self.Bifusion0 = BiFusion([ch_list[3], ch_list[5]], ch_list[5])
        self.Rep_p4 = BepC3Layer(
            ch_list[5], ch_list[5], num_repeats[5], csp_e, block=block, act=act)

        # Rep_p3
        self.reduce_layer1 = SimConv(ch_list[5], ch_list[6], 1, 1)
        self.Bifusion1 = BiFusion([ch_list[5], ch_list[6]], ch_list[6])
        self.Rep_p3 = BepC3Layer(
            ch_list[6], ch_list[6], num_repeats[6], csp_e, block=block, act=act)

        # Rep_n3
        self.downsample2 = SimConv(ch_list[6], ch_list[7], 3, 2)
        self.Rep_n3 = BepC3Layer(
            ch_list[6] + ch_list[7],
            ch_list[8],
            num_repeats[7],
            csp_e,
            block=block,
            act=act)

        # Rep_n4
        self.downsample1 = SimConv(ch_list[8], ch_list[9], 3, 2)
        self.Rep_n4 = BepC3Layer(
            ch_list[5] + ch_list[9],
            ch_list[10],
            num_repeats[8],
            csp_e,
            block=block,
            act=act)

    def forward(self, feats, for_mot=False):
        assert len(feats) == len(self.in_channels)
        [x3, x2, x1, x0] = feats  # p2, p3, p4, p5 

        # top-down FPN
        fpn_out0 = self.reduce_layer0(x0)
        f_concat_layer0 = self.Bifusion0([fpn_out0, x1, x2])
        f_out0 = self.Rep_p4(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(f_out0)
        f_concat_layer1 = self.Bifusion1([fpn_out1, x2, x3])
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
