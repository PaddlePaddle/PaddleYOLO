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

import math
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from paddle.nn.initializer import Constant
from ppdet.modeling.initializer import conv_init_, normal_
from ppdet.core.workspace import register, serializable
from ..shape_spec import ShapeSpec

__all__ = [
    'RepConv', 'RepLayer', 'BepC3Layer', 'SimSPPF', 'EfficientRep',
    'CSPBepBackbone'
]


def get_activation(name="silu"):
    if name == "silu":
        module = nn.Silu()
    elif name == "relu":
        module = nn.ReLU()
    elif name in ["LeakyReLU", 'leakyrelu', 'lrelu']:
        module = nn.LeakyReLU(0.1)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class SiLU(nn.Layer):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return x * F.sigmoid(x)


class BaseConv(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 groups=1,
                 bias=False,
                 act="silu"):
        super(BaseConv, self).__init__()
        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=(ksize - 1) // 2,
            groups=groups,
            bias_attr=bias)
        self.bn = nn.BatchNorm2D(
            out_channels,
            epsilon=1e-3,  # for amp(fp16)
            momentum=0.97,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self.act = get_activation(act)  # silu
        self._init_weights()

    def _init_weights(self):
        conv_init_(self.conv)

    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.training:
            y = self.act(x)
        else:
            if isinstance(self.act, nn.Silu):
                self.act = SiLU()
            y = self.act(x)
        return y


class RepConv(nn.Layer):
    """
    RepVGG Conv BN Relu Block, see https://arxiv.org/abs/2101.03697
    named RepVGGBlock in YOLOv6
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 groups=1,
                 act='relu',
                 deploy=False):
        super(RepConv, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert kernel_size == 3
        assert padding == 1
        padding_11 = padding - kernel_size // 2
        self.stride = stride  # not always 1

        self.act = nn.ReLU()  # get_activation(act)

        if self.deploy:
            self.rbr_reparam = nn.Conv2D(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias_attr=True)
        else:
            self.rbr_identity = (nn.BatchNorm2D(
                in_channels, epsilon=1e-3, momentum=0.97)
                                 if out_channels == in_channels and stride == 1
                                 else None)
            self.rbr_dense = nn.Sequential(* [
                nn.Conv2D(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    groups=groups,
                    bias_attr=False),
                nn.BatchNorm2D(out_channels),
            ])
            self.rbr_1x1 = nn.Sequential(* [
                nn.Conv2D(
                    in_channels,
                    out_channels,
                    1,
                    stride,
                    padding_11,
                    groups=groups,
                    bias_attr=False),
                nn.BatchNorm2D(out_channels),
            ])

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            x = self.rbr_reparam(inputs)
            y = self.act(x)
            return y

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        x = self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out
        y = self.act(x)
        return y

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid, )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1]._mean
            running_var = branch[1]._variance
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1]._epsilon
        else:
            assert isinstance(branch, nn.BatchNorm2D)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = paddle.zeros([self.in_channels, input_dim, 3, 3])

                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch._mean
            running_var = branch._variance
            gamma = branch.weight
            beta = branch.bias
            eps = branch._epsilon
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape((-1, 1, 1, 1))
        return kernel * t, beta - running_mean * gamma / std

    def convert_to_deploy(self):
        if hasattr(self, "rbr_reparam"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2D(
            self.in_channels,
            self.out_channels,
            3,
            self.stride,
            padding=1,
            groups=self.groups,
            bias_attr=True)
        self.rbr_reparam.weight.set_value(kernel)
        self.rbr_reparam.bias.set_value(bias)
        self.__delattr__("rbr_dense")
        self.__delattr__("rbr_1x1")
        if hasattr(self, "rbr_identity"):
            self.__delattr__("rbr_identity")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")
        self.deploy = True


class RepLayer(nn.Layer):
    """
    RepLayer with RepConvs, like CSPLayer(C3) in YOLOv5/YOLOX
    named RepBlock in YOLOv6
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_repeats=1,
                 block=RepConv,
                 basic_block=RepConv):
        super(RepLayer, self).__init__()
        if block == RepConv:
            # in n/t/s
            self.conv1 = block(in_channels, out_channels)
            self.block = (nn.Sequential(*(block(out_channels, out_channels)
                                          for _ in range(num_repeats - 1)))
                          if num_repeats > 1 else None)

        elif block == BottleRep:
            # in m/l
            self.conv1 = BottleRep(
                in_channels, out_channels, basic_block=basic_block, alpha=True)
            num_repeats = num_repeats // 2
            self.block = nn.Sequential(*(BottleRep(
                out_channels, out_channels, basic_block=basic_block, alpha=True
            ) for _ in range(num_repeats - 1))) if num_repeats > 1 else None

    def forward(self, x):
        x = self.conv1(x)
        if self.block is not None:
            x = self.block(x)
        return x


class BottleRep(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 basic_block=RepConv,
                 alpha=False):
        super(BottleRep, self).__init__()
        self.conv1 = basic_block(in_channels, out_channels)
        self.conv2 = basic_block(out_channels, out_channels)
        if in_channels != out_channels:
            self.shortcut = False
        else:
            self.shortcut = True
        if alpha:
            self.alpha = self.create_parameter(
                shape=[1],
                attr=ParamAttr(initializer=Constant(value=1.)),
                dtype="float32")
        else:
            self.alpha = 1.0

    def forward(self, x):
        outputs = self.conv1(x)
        outputs = self.conv2(outputs)
        return outputs + self.alpha * x if self.shortcut else outputs


class BepC3Layer(nn.Layer):
    '''Beer-mug RepC3 Block
       named BepC3 in YOLOv6
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_repeats=1,
                 csp_e=0.5,
                 block=RepConv,
                 act='silu'):
        super(BepC3Layer, self).__init__()
        c_ = int(out_channels * csp_e)  # hidden channels
        self.cv1 = BaseConv(in_channels, c_, 1, 1, act="relu")
        self.cv2 = BaseConv(in_channels, c_, 1, 1, act="relu")
        self.cv3 = BaseConv(2 * c_, out_channels, 1, 1, act="relu")
        if act == 'silu':
            self.cv1 = BaseConv(in_channels, c_, 1, 1)
            self.cv2 = BaseConv(in_channels, c_, 1, 1)
            self.cv3 = BaseConv(2 * c_, out_channels, 1, 1)
        self.m = RepLayer(
            c_, c_, num_repeats, block=BottleRep, basic_block=block)

    def forward(self, x):
        return self.cv3(paddle.concat((self.m(self.cv1(x)), self.cv2(x)), 1))


class SimConv(nn.Layer):
    """Simplified Conv BN ReLU"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 groups=1,
                 bias=False):
        super(SimConv, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias_attr=bias)
        self.bn = nn.BatchNorm2D(
            out_channels,
            epsilon=1e-3,
            momentum=0.97,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self.act = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        conv_init_(self.conv)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SimSPPF(nn.Layer):
    """Simplified SPPF with SimConv, use relu"""

    def __init__(self, in_channels, out_channels, kernel_size=5):
        super(SimSPPF, self).__init__()
        hidden_channels = in_channels // 2
        self.conv1 = SimConv(in_channels, hidden_channels, 1, 1)
        self.mp = nn.MaxPool2D(
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.conv2 = SimConv(hidden_channels * 4, out_channels, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.mp(x)
        y2 = self.mp(y1)
        y3 = self.mp(y2)
        concats = paddle.concat([x, y1, y2, y3], 1)
        return self.conv2(concats)


class SPPF(nn.Layer):
    """SPPF with BaseConv, use silu"""

    def __init__(self, in_channels, out_channels, kernel_size=5, act='silu'):
        super(SPPF, self).__init__()
        hidden_channels = in_channels // 2
        self.cv1 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, act=act)
        self.conv2 = BaseConv(
            hidden_channels * 4, out_channels, ksize=1, stride=1, act=act)
        self.mp = nn.MaxPool2D(
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.mp(x)
        y2 = self.mp(y1)
        y3 = self.mp(y2)
        concats = paddle.concat([x, y1, y2, y3], 1)
        return self.conv2(concats)


class Transpose(nn.Layer):
    '''Normal Transpose, default for upsampling'''

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.upsample_transpose = nn.Conv2DTranspose(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias_attr=True)

    def forward(self, x):
        return self.upsample_transpose(x)


def make_divisible(x, divisor):
    return math.ceil(x / divisor) * divisor


@register
@serializable
class EfficientRep(nn.Layer):
    """EfficientRep backbone of YOLOv6 n/t/s """
    __shared__ = ['width_mult', 'depth_mult', 'trt', 'act', 'training_mode']

    def __init__(self,
                 width_mult=1.0,
                 depth_mult=1.0,
                 num_repeats=[1, 6, 12, 18, 6],
                 channels_list=[64, 128, 256, 512, 1024],
                 return_idx=[2, 3, 4],
                 training_mode='repvgg',
                 act='relu',
                 trt=False):
        super(EfficientRep, self).__init__()
        num_repeats = [(max(round(i * depth_mult), 1) if i > 1 else i)
                       for i in (num_repeats)]
        channels_list = [
            make_divisible(i * width_mult, 8) for i in (channels_list)
        ]
        self.return_idx = return_idx
        self._out_channels = [channels_list[i] for i in self.return_idx]
        self.strides = [[2, 4, 8, 16, 32, 64][i] for i in self.return_idx]

        block = get_block(training_mode)
        self.stem = block(3, channels_list[0], 3, 2)
        self.blocks = []
        for i, (out_ch,
                num_repeat) in enumerate(zip(channels_list, num_repeats)):
            if i == 0: continue
            in_ch = channels_list[i - 1]
            stage = []

            repconv = self.add_sublayer('stage{}.repconv'.format(i + 1),
                                        block(in_ch, out_ch, 3, 2))
            stage.append(repconv)

            replayer = self.add_sublayer(
                'stage{}.replayer'.format(i + 1),
                RepLayer(
                    out_ch, out_ch, num_repeat, block=block))
            stage.append(replayer)

            if i == len(channels_list) - 1:
                simsppf_layer = self.add_sublayer(
                    'stage{}.simsppf'.format(i + 1),
                    SimSPPF(
                        out_ch, out_ch, kernel_size=5))
                stage.append(simsppf_layer)
            self.blocks.append(nn.Sequential(*stage))

    def forward(self, inputs):
        x = inputs['image']
        outputs = []
        x = self.stem(x)
        for i, layer in enumerate(self.blocks):
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


@register
@serializable
class CSPBepBackbone(nn.Layer):
    """CSPBepBackbone of YOLOv6 m/l """
    __shared__ = ['width_mult', 'depth_mult', 'trt', 'act', 'training_mode']

    def __init__(self,
                 width_mult=1.0,
                 depth_mult=1.0,
                 num_repeats=[1, 6, 12, 18, 6],
                 channels_list=[64, 128, 256, 512, 1024],
                 return_idx=[2, 3, 4],
                 csp_e=0.5,
                 training_mode='repvgg',
                 act='relu',
                 trt=False):
        super(CSPBepBackbone, self).__init__()
        num_repeats = [(max(round(i * depth_mult), 1) if i > 1 else i)
                       for i in (num_repeats)]
        channels_list = [
            make_divisible(i * width_mult, 8) for i in (channels_list)
        ]
        self.return_idx = return_idx
        self._out_channels = [channels_list[i] for i in self.return_idx]
        self.strides = [[2, 4, 8, 16, 32, 64][i] for i in self.return_idx]

        block = get_block(training_mode)
        self.stem = block(3, channels_list[0], 3, 2)
        self.blocks = []
        if csp_e == 0.67: csp_e = float(2) / 3
        for i, (out_ch,
                num_repeat) in enumerate(zip(channels_list, num_repeats)):
            if i == 0: continue
            in_ch = channels_list[i - 1]
            stage = []

            repconv = self.add_sublayer('stage{}.repconv'.format(i + 1),
                                        block(in_ch, out_ch, 3, 2))
            stage.append(repconv)

            bepc3layer = self.add_sublayer(
                'stage{}.bepc3layer'.format(i + 1),
                BepC3Layer(
                    out_ch,
                    out_ch,
                    num_repeat,
                    csp_e=csp_e,
                    block=block,
                    act=act))
            stage.append(bepc3layer)

            if i == len(channels_list) - 1:
                if training_mode == 'conv_silu':
                    sppf_layer = self.add_sublayer(
                        'stage{}.sppf'.format(i + 1),
                        SPPF(
                            out_ch, out_ch, kernel_size=5, act='silu'))
                    stage.append(sppf_layer)
                else:
                    simsppf_layer = self.add_sublayer(
                        'stage{}.simsppf'.format(i + 1),
                        SimSPPF(
                            out_ch, out_ch, kernel_size=5))
                    stage.append(simsppf_layer)
            self.blocks.append(nn.Sequential(*stage))

    def forward(self, inputs):
        x = inputs['image']
        outputs = []
        x = self.stem(x)
        for i, layer in enumerate(self.blocks):
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


def get_block(mode):
    if mode == 'repvgg':
        return RepConv
    elif mode == 'conv_relu':
        return SimConvWrapper
    elif mode == 'conv_silu':
        return ConvWrapper
    else:
        raise NotImplementedError("Undefied Repblock choice for mode {}".format(
            mode))


class ConvWrapper(nn.Layer):
    '''Wrapper for normal Conv with SiLU activation'''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 groups=1,
                 bias=True):
        super().__init__()
        self.base_block = BaseConv(in_channels, out_channels, kernel_size,
                                   stride, groups, bias)

    def forward(self, x):
        return self.base_block(x)


class SimConvWrapper(nn.Layer):
    '''Wrapper for normal Conv with ReLU activation'''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 groups=1,
                 bias=True):
        super().__init__()
        self.base_block = SimConv(in_channels, out_channels, kernel_size,
                                  stride, groups, bias)

    def forward(self, x):
        return self.base_block(x)
