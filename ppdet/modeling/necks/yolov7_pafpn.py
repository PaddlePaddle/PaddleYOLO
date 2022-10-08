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
from ppdet.core.workspace import register, serializable
from ..backbones.yolov7_elannet import BaseConv, ELANLayer, ELAN2Layer, MPConvLayer, RepConv, DownC
from ..shape_spec import ShapeSpec

__all__ = ['ELANFPN', 'ELANFPNP6']


@register
@serializable
class ELANFPN(nn.Layer):
    """
    YOLOv7 E-ELAN FPN, used in P5 model like ['tiny', 'L', 'X'], return 3 feats
    """
    __shared__ = ['arch', 'depth_mult', 'width_mult', 'act', 'trt']

    # [in_ch, mid_ch1, mid_ch2, out_ch] of each ELANLayer (2 FPN + 2 PAN): 
    ch_settings = {
        'tiny': [[256, 64, 64, 128], [128, 32, 32, 64], [64, 64, 64, 128],
                 [128, 128, 128, 256]],
        'L': [[512, 256, 128, 256], [256, 128, 64, 128], [128, 256, 128, 256],
              [256, 512, 256, 512]],
        'X': [[640, 256, 256, 320], [320, 128, 128, 160], [160, 256, 256, 320],
              [320, 512, 512, 640]],
    }
    # concat_list of each ELANLayer:
    concat_list_settings = {
        'tiny': [-1, -2, -3, -4],
        'L': [-1, -2, -3, -4, -5, -6],
        'X': [-1, -3, -5, -7, -8],
    }
    num_blocks = {'tiny': 2, 'L': 4, 'X': 6}

    def __init__(
            self,
            arch='L',
            depth_mult=1.0,
            width_mult=1.0,
            in_channels=[512, 1024, 512],  # layer num: 24 37 51 [c3,c4,c5]
            out_channels=[256, 512, 1024],  # layer num: 75 88 101
            depthwise=False,
            act='silu',
            trt=False):
        super(ELANFPN, self).__init__()
        self.in_channels = in_channels
        self.arch = arch
        concat_list = self.concat_list_settings[arch]
        num_blocks = self.num_blocks[arch]
        ch_settings = self.ch_settings[arch]
        self._out_channels = [chs[-1] * 2 for chs in ch_settings[1:]]

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        in_ch, mid_ch1, mid_ch2, out_ch = ch_settings[0][:]
        self.lateral_conv1 = BaseConv(
            self.in_channels[2], out_ch, 1, 1, act=act)  # 512->256
        self.route_conv1 = BaseConv(
            self.in_channels[1], out_ch, 1, 1, act=act)  # 1024->256
        self.elan_fpn1 = ELANLayer(
            out_ch * 2,
            mid_ch1,
            mid_ch2,
            out_ch,
            num_blocks,
            concat_list,
            depthwise,
            act=act)

        in_ch, mid_ch1, mid_ch2, out_ch = ch_settings[1][:]
        self.lateral_conv2 = BaseConv(in_ch, out_ch, 1, 1, act=act)  # 256->128
        self.route_conv2 = BaseConv(
            self.in_channels[0], out_ch, 1, 1, act=act)  # 512->128
        self.elan_fpn2 = ELANLayer(
            out_ch * 2,
            mid_ch1,
            mid_ch2,
            out_ch,
            num_blocks,
            concat_list,
            depthwise,
            act=act)

        in_ch, mid_ch1, mid_ch2, out_ch = ch_settings[2][:]
        if self.arch in ['L', 'X']:
            self.mp_conv1 = MPConvLayer(in_ch, out_ch, 0.5, depthwise, act=act)
            # TODO: named down_conv1
        elif self.arch in ['tiny']:
            self.mp_conv1 = BaseConv(in_ch, out_ch, 3, 2, act=act)
        else:
            raise AttributeError("Unsupported arch type: {}".format(self.arch))
        self.elan_pan1 = ELANLayer(
            out_ch * 2,
            mid_ch1,
            mid_ch2,
            out_ch,
            num_blocks,
            concat_list,
            depthwise,
            act=act)

        in_ch, mid_ch1, mid_ch2, out_ch = ch_settings[3][:]
        if self.arch in ['L', 'X']:
            self.mp_conv2 = MPConvLayer(in_ch, out_ch, 0.5, depthwise, act=act)
        elif self.arch in ['tiny']:
            self.mp_conv2 = BaseConv(in_ch, out_ch, 3, 2, act=act)
        else:
            raise AttributeError("Unsupported arch type: {}".format(self.arch))
        self.elan_pan2 = ELANLayer(
            out_ch + self.in_channels[2],  # concat([pan_out1_down, c5], 1)
            mid_ch1,
            mid_ch2,
            out_ch,
            num_blocks,
            concat_list,
            depthwise,
            act=act)

        self.repconvs = nn.LayerList()
        Conv = RepConv if self.arch == 'L' else BaseConv
        for out_ch in self._out_channels:
            self.repconvs.append(Conv(int(out_ch // 2), out_ch, 3, 1, act=act))

    def forward(self, feats, for_mot=False):
        assert len(feats) == len(self.in_channels)
        [c3, c4, c5] = feats  # 24  37  51
        # [8, 512, 80, 80] [8, 1024, 40, 40] [8, 512, 20, 20]

        # Top-Down FPN
        p5_lateral = self.lateral_conv1(c5)  # 512->256
        p5_up = self.upsample(p5_lateral)
        route_c4 = self.route_conv1(c4)  # 1024->256 # route
        f_out1 = paddle.concat([route_c4, p5_up], 1)  # 512 # [8, 512, 40, 40]
        fpn_out1 = self.elan_fpn1(f_out1)  # 512 -> 128*4 + 256*2 -> 1024 -> 256
        # 63

        fpn_out1_lateral = self.lateral_conv2(fpn_out1)  # 256->128
        fpn_out1_up = self.upsample(fpn_out1_lateral)
        route_c3 = self.route_conv2(c3)  # 512->128 # route
        f_out2 = paddle.concat([route_c3, fpn_out1_up], 1)  # 256
        fpn_out2 = self.elan_fpn2(f_out2)  # 256 -> 64*4 + 128*2 -> 512 -> 128
        # layer 75: [8, 128, 80, 80]

        # Buttom-Up PAN
        p_out1_down = self.mp_conv1(fpn_out2)  # 128
        p_out1 = paddle.concat([p_out1_down, fpn_out1], 1)  # 128*2 + 256 -> 512
        pan_out1 = self.elan_pan1(p_out1)  # 512 -> 128*4 + 256*2 -> 1024 -> 256
        # layer 88: [8, 256, 40, 40]

        pan_out1_down = self.mp_conv2(pan_out1)  # 256
        p_out2 = paddle.concat([pan_out1_down, c5], 1)  # 256*2 + 512 -> 1024
        pan_out2 = self.elan_pan2(
            p_out2)  # 1024 -> 256*4 + 512*2 -> 2048 -> 512
        # layer 101: [8, 512, 20, 20]

        outputs = []
        pan_outs = [fpn_out2, pan_out1, pan_out2]  # 75 88 101
        for i, out in enumerate(pan_outs):
            outputs.append(self.repconvs[i](out))
        return outputs

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    @property
    def out_shape(self):
        return [ShapeSpec(channels=c) for c in self._out_channels]


@register
@serializable
class ELANFPNP6(nn.Layer):
    """
    YOLOv7P6 E-ELAN FPN, used in P6 model like ['W6', 'E6', 'D6', 'E6E']
    return 4 feats
    """
    __shared__ = ['arch', 'depth_mult', 'width_mult', 'act', 'use_aux', 'trt']

    # in_ch, mid_ch1, mid_ch2, out_ch of each ELANLayer (3 FPN + 3 PAN): 
    ch_settings = {
        'W6':
        [[512, 384, 192, 384], [384, 256, 128, 256], [256, 128, 64, 128],
         [128, 256, 128, 256], [256, 384, 192, 384], [384, 512, 256, 512]],
        'E6': [[640, 384, 192, 480], [480, 256, 128, 320], [320, 128, 64, 160],
               [160, 256, 128, 320], [320, 384, 192, 480],
               [480, 512, 256, 640]],
        'D6': [[768, 384, 192, 576], [576, 256, 128, 384], [384, 128, 64, 192],
               [192, 256, 128, 384], [384, 384, 192, 576],
               [576, 512, 256, 768]],
        'E6E': [[640, 384, 192, 480], [480, 256, 128, 320],
                [320, 128, 64, 160], [160, 256, 128, 320],
                [320, 384, 192, 480], [480, 512, 256, 640]],
    }
    # concat_list of each ELANLayer:
    concat_list_settings = {
        'W6': [-1, -2, -3, -4, -5, -6],
        'E6': [-1, -2, -3, -4, -5, -6, -7, -8],
        'D6': [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10],
        'E6E': [-1, -2, -3, -4, -5, -6, -7, -8],
    }
    num_blocks = {'W6': 4, 'E6': 6, 'D6': 8, 'E6E': 6}

    def __init__(
            self,
            arch='W6',
            use_aux=False,
            depth_mult=1.0,
            width_mult=1.0,
            in_channels=[256, 512, 768, 512],  # 19 28 37 47 (c3,c4,c5,c6)
            out_channels=[256, 512, 768, 1024],  # layer: 83 93 103 113
            depthwise=False,
            act='silu',
            trt=False):
        super(ELANFPNP6, self).__init__()
        self.in_channels = in_channels
        self.arch = arch
        self.use_aux = use_aux
        concat_list = self.concat_list_settings[arch]
        num_blocks = self.num_blocks[arch]
        ch_settings = self.ch_settings[arch]
        self._out_channels = [chs[-1] * 2 for chs in ch_settings[2:]]
        if self.training and self.use_aux:
            chs_aux = [chs[-1] for chs in ch_settings[:3][::-1]
                       ] + [self.in_channels[3]]
            self.in_channels_aux = chs_aux
            self._out_channels = self._out_channels + [320, 640, 960, 1280]
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        ELANBlock = ELAN2Layer if self.arch in ['E6E'] else ELANLayer

        in_ch, mid_ch1, mid_ch2, out_ch = ch_settings[0][:]
        self.lateral_conv1 = BaseConv(
            self.in_channels[3], out_ch, 1, 1, act=act)  # 512->384
        self.route_conv1 = BaseConv(
            self.in_channels[2], out_ch, 1, 1, act=act)  # 768->384
        self.elan_fpn1 = ELANBlock(
            out_ch * 2,
            mid_ch1,
            mid_ch2,
            out_ch,
            num_blocks,
            concat_list,
            depthwise,
            act=act)

        in_ch, mid_ch1, mid_ch2, out_ch = ch_settings[1][:]
        self.lateral_conv2 = BaseConv(in_ch, out_ch, 1, 1, act=act)  # 384->256
        self.route_conv2 = BaseConv(
            self.in_channels[1], out_ch, 1, 1, act=act)  # 512->256
        self.elan_fpn2 = ELANBlock(
            out_ch * 2,
            mid_ch1,
            mid_ch2,
            out_ch,
            num_blocks,
            concat_list,
            depthwise,
            act=act)

        in_ch, mid_ch1, mid_ch2, out_ch = ch_settings[2][:]
        self.lateral_conv3 = BaseConv(in_ch, out_ch, 1, 1, act=act)  # 256->128
        self.route_conv3 = BaseConv(
            self.in_channels[0], out_ch, 1, 1, act=act)  # 256->128
        self.elan_fpn3 = ELANBlock(
            out_ch * 2,
            mid_ch1,
            mid_ch2,
            out_ch,
            num_blocks,
            concat_list,
            depthwise,
            act=act)

        in_ch, mid_ch1, mid_ch2, out_ch = ch_settings[3][:]
        if self.arch in ['W6']:
            self.down_conv1 = BaseConv(in_ch, out_ch, 3, 2, act=act)
        elif self.arch in ['E6', 'D6', 'E6E']:
            self.down_conv1 = DownC(in_ch, out_ch, 2, act=act)
        else:
            raise AttributeError("Unsupported arch type: {}".format(self.arch))
        self.elan_pan1 = ELANBlock(
            out_ch * 2,
            mid_ch1,
            mid_ch2,
            out_ch,
            num_blocks,
            concat_list,
            depthwise,
            act=act)

        in_ch, mid_ch1, mid_ch2, out_ch = ch_settings[4][:]
        if self.arch in ['W6']:
            self.down_conv2 = BaseConv(in_ch, out_ch, 3, 2, act=act)
        elif self.arch in ['E6', 'D6', 'E6E']:
            self.down_conv2 = DownC(in_ch, out_ch, 2, act=act)
        else:
            raise AttributeError("Unsupported arch type: {}".format(self.arch))
        self.elan_pan2 = ELANBlock(
            out_ch * 2,
            mid_ch1,
            mid_ch2,
            out_ch,
            num_blocks,
            concat_list,
            depthwise,
            act=act)

        in_ch, mid_ch1, mid_ch2, out_ch = ch_settings[5][:]
        if self.arch in ['W6']:
            self.down_conv3 = BaseConv(in_ch, out_ch, 3, 2, act=act)
        elif self.arch in ['E6', 'D6', 'E6E']:
            self.down_conv3 = DownC(in_ch, out_ch, 2, act=act)
        else:
            raise AttributeError("Unsupported arch type: {}".format(self.arch))
        self.elan_pan3 = ELANBlock(
            out_ch + self.in_channels[3],  # concat([pan_out2_down, c6], 1)
            mid_ch1,
            mid_ch2,
            out_ch,
            num_blocks,
            concat_list,
            depthwise,
            act=act)

        self.repconvs = nn.LayerList()
        Conv = RepConv if self.arch == 'L' else BaseConv
        for i, _out_ch in enumerate(self._out_channels[:4]):
            self.repconvs.append(Conv(_out_ch // 2, _out_ch, 3, 1, act=act))

        if self.training and self.use_aux:
            self.repconvs_aux = nn.LayerList()
            for i, _out_ch in enumerate(self._out_channels[4:]):
                self.repconvs_aux.append(
                    Conv(
                        self.in_channels_aux[i], _out_ch, 3, 1, act=act))

    def forward(self, feats, for_mot=False):
        assert len(feats) == len(self.in_channels)
        [c3, c4, c5, c6] = feats  # 19 28 37 47
        # [8, 256, 160, 160] [8, 512, 80, 80] [8, 768, 40, 40] [8, 512, 20, 20]

        # Top-Down FPN
        p6_lateral = self.lateral_conv1(c6)  # 512->384
        p6_up = self.upsample(p6_lateral)
        route_c5 = self.route_conv1(c5)  # 768->384 # route
        f_out1 = paddle.concat([route_c5, p6_up], 1)  # 768 # [8, 768, 40, 40]
        fpn_out1 = self.elan_fpn1(f_out1)  # 768 -> 192*4 + 384*2 -> 1536 -> 384
        # layer 59: [8, 384, 40, 40]

        fpn_out1_lateral = self.lateral_conv2(fpn_out1)  # 384->256
        fpn_out1_up = self.upsample(fpn_out1_lateral)
        route_c4 = self.route_conv2(c4)  # 512->256 # route
        f_out2 = paddle.concat([route_c4, fpn_out1_up],
                               1)  # 512 # [8, 512, 80, 80]
        fpn_out2 = self.elan_fpn2(f_out2)  # 512 -> 128*4 + 256*2 -> 1024 -> 256
        # layer 71: [8, 256, 80, 80]

        fpn_out2_lateral = self.lateral_conv3(fpn_out2)  # 256->128
        fpn_out2_up = self.upsample(fpn_out2_lateral)
        route_c3 = self.route_conv3(c3)  # 512->128 # route
        f_out3 = paddle.concat([route_c3, fpn_out2_up], 1)  # 256
        fpn_out3 = self.elan_fpn3(f_out3)  # 256 -> 64*4 + 128*2 -> 512 -> 128
        # layer 83: [8, 128, 160, 160]

        # Buttom-Up PAN
        p_out1_down = self.down_conv1(fpn_out3)  # 128->256
        p_out1 = paddle.concat([p_out1_down, fpn_out2], 1)  # 256 + 256 -> 512
        pan_out1 = self.elan_pan1(p_out1)  # 512 -> 128*4 + 256*2 -> 1024 -> 256
        # layer 93: [8, 256, 80, 80]

        pan_out1_down = self.down_conv2(pan_out1)  # 256->384
        p_out2 = paddle.concat([pan_out1_down, fpn_out1], 1)  # 384 + 384 -> 768
        pan_out2 = self.elan_pan2(p_out2)  # 768 -> 192*4 + 384*2 -> 1536 -> 384
        # layer 103: [8, 384, 40, 40]

        pan_out2_down = self.down_conv3(pan_out2)  # 384->512
        p_out3 = paddle.concat([pan_out2_down, c6], 1)  # 512 + 512 -> 1024
        pan_out3 = self.elan_pan3(
            p_out3)  # 1024 -> 256*4 + 512*2 -> 2048 -> 512
        # layer 113: [8, 512, 20, 20]

        outputs = []
        pan_outs = [fpn_out3, pan_out1, pan_out2, pan_out3]  # 83 93 103 113
        for i, out in enumerate(pan_outs):
            outputs.append(self.repconvs[i](out))

        if self.training and self.use_aux:
            aux_outs = [fpn_out3, fpn_out2, fpn_out1, c6]  # 83 71 59 47
            for i, out in enumerate(aux_outs):
                outputs.append(self.repconvs_aux[i](out))
        return outputs

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    @property
    def out_shape(self):
        return [ShapeSpec(channels=c) for c in self._out_channels]
