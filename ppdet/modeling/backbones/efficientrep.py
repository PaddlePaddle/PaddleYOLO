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

__all__ = ['RepConv', 'RepLayer', 'SimSPPF', 'EfficientRep']


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
        # self.act = get_activation(act)
        self._init_weights()

    def _init_weights(self):
        conv_init_(self.conv)

    def forward(self, x):
        x = self.bn(self.conv(x))
        # y = self.act(x)
        y = x * F.sigmoid(x)
        return y


class RepLayer(nn.Layer):
    """
    RepLayer with RepConvs, like CSPLayer(C3) in YOLOv5/YOLOX
    """

    def __init__(self, in_channels, out_channels, num_repeats=1):
        super().__init__()
        self.conv1 = RepConv(in_channels, out_channels)
        self.blocks = (nn.Sequential(*(RepConv(out_channels, out_channels)
                                       for _ in range(num_repeats - 1)))
                       if num_repeats > 1 else None)

    def forward(self, x):
        x = self.conv1(x)
        if self.blocks is not None:
            x = self.blocks(x)
        return x


class RepConv(nn.Layer):
    # RepVGG Conv BN Block, see https://arxiv.org/abs/2101.03697
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

        self.act = get_activation(act)

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
    """Simplified SPPF with SimConv"""

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
    """EfficientRep backbone of MT-YOLOv6"""
    __shared__ = ['width_mult', 'depth_mult', 'trt']

    def __init__(self,
                 width_mult=1.0,
                 depth_mult=1.0,
                 num_repeats=[1, 6, 12, 18, 6],
                 channels_list=[64, 128, 256, 512, 1024],
                 return_idx=[2, 3, 4],
                 depthwise=False,
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

        self.stem = RepConv(3, channels_list[0], 3, 2)
        self.blocks = []
        for i, (out_ch,
                num_repeat) in enumerate(zip(channels_list, num_repeats)):
            if i == 0: continue
            in_ch = channels_list[i - 1]
            stage = []

            repconv = self.add_sublayer('stage{}.repconv'.format(i + 1),
                                        RepConv(in_ch, out_ch, 3, 2))
            stage.append(repconv)

            replayer = self.add_sublayer('stage{}.replayer'.format(i + 1),
                                         RepLayer(out_ch, out_ch, num_repeat))
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
