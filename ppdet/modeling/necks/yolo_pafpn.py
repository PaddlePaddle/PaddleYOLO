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
from ppdet.modeling.layers import DropBlock
from ppdet.modeling.ops import get_act_fn
from ..backbones.darknet import ConvBNLayer
from ..shape_spec import ShapeSpec
from ..backbones.csp_darknet import BaseConv, DWConv, CSPLayer, ELANLayer, ELAN2Layer, MPConvLayer, RepConv, DownC
from .ASFF import ASFF

@register
class YOLOPAFPN(nn.Layer):
    def __init__(self,
                 depth=1.0,
                 width=1.0,
                 backbone='CSPDarknet',
                 neck_type='yolo',
                 neck_mode='all',
                 in_features=('dark3', 'dark4', 'dark5'),
                 in_channels=[256, 512, 1024],
                 depthwise=False,
                 act='silu',
                 use_att=None,
                 asff_channel=2,
                 expand_kernel=3):
        super(YOLOPAFPN, self).__init__()
        Conv = DWConv if depthwise else BaseConv
        self.lateral_conv0 = BaseConv(int(in_channels[2] * width),int(in_channels[1] * width),1,1,act=act)
        self.upsample=nn.Upsample(scale_factor=2,mode="nearest")
        self.lateral_conv0 = BaseConv(
                int(in_channels[2] * width),
                int(in_channels[1] * width),
                1,
                1,
                act=act)
        self.C3_p4 = CSPLayer(
                int(2 * in_channels[1] * width),
                int(in_channels[1] * width),
                round(3 * depth),
                False,
                depthwise=depthwise,
                act=act)  # cat
        self.reduce_conv1 = BaseConv(
                int(in_channels[1] * width),
                int(in_channels[0] * width),
                1,
                1,
                act=act)
        self.C3_p3 = CSPLayer(
                int(2 * in_channels[0] * width),
                int(in_channels[0] * width),
                round(3 * depth),
                False,
                depthwise=depthwise,
                act=act)

            # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width),
            int(in_channels[0] * width),
            3,
            2,
            act=act)
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act)

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width),
            int(in_channels[1] * width),
            3,
            2,
            act=act)
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act)
        self.use_att = use_att
        default_attention_list = ['ASFF', 'ASFF_sim']
        if use_att is not None and use_att not in default_attention_list:
            logging.warning(
                'YOLOX-PAI backbone must in [ASFF, ASFF_sim], otherwise we use ASFF as default'
            )
        if self.use_att == 'ASFF' or self.use_att == 'ASFF_sim':
            self.asff_1=ASFF(
                level=0,
                type=self.use_att,
                asff_channel=asff_channel,
                expand_kernel=expand_kernel,
                multiplier=width,
                act=act)
            self.asff_2=ASFF(
                level=1,
                type=self.use_att,
                asff_channel=asff_channel,
                expand_kernel=expand_kernel,
                multiplier=width,
                act=act)
            self.asff_3=ASFF(
                level=2,
                type=self.use_att,
                asff_channel=asff_channel,
                expand_kernel=expand_kernel,
                multiplier=width,
                act=act)

    def forward(self,inputs):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """
        [x2, x1, x0] = inputs
        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 =  paddle.concat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = paddle.concat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = paddle.concat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = paddle.concat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32
        outputs = (pan_out2, pan_out1, pan_out0)
        if self.use_att == 'ASFF' or self.use_att == 'ASFF_sim':
            pan_out0 = self.asff_1(outputs)
            pan_out1 = self.asff_2(outputs)
            pan_out2 = self.asff_3(outputs)
            outputs = (pan_out2, pan_out1, pan_out0)
        return outputs
        '''
        inputs= torch.Size([8, 128, 80, 80])
                torch.Size([8, 256, 40, 40])
                torch.Size([8, 512, 20, 20])
        outputs=torch.Size([8, 128, 80, 80])
                torch.Size([8, 256, 40, 40])
                torch.Size([8, 512, 20, 20])

        '''
