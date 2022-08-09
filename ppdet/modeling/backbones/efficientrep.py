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
from ppdet.core.workspace import register, serializable
from ..shape_spec import ShapeSpec

__all__ = ['RepVGGBlock', 'RepBlock', 'SimSPPF']


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


class Conv(nn.Layer):
    '''Normal Conv with SiLU activation'''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 groups=1,
                 bias=False,
                 act='silu'):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias_attr=bias)
        self.bn = nn.BatchNorm2D(out_channels, epsilon=1e-3, momentum=0.97)
        # self.act = get_activation(act)

    def forward(self, x):
        x = self.bn(self.conv(x))
        # y = self.act(x)
        y = x * F.sigmoid(x)
        return y


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    """Basic cell for rep-style block, including conv and bn"""
    result = nn.Sequential()
    result.add_sublayer(
        "conv",
        nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias_attr=False))
    result.add_sublayer(
        "bn",
        nn.BatchNorm2D(
            num_features=out_channels, epsilon=1e-3, momentum=0.97))
    return result


class RepBlock(nn.Layer):
    """
    RepBlock is a stage block with rep-style basic block
    """

    def __init__(self, in_channels, out_channels, n=1):
        super().__init__()
        self.conv1 = RepVGGBlock(in_channels, out_channels)
        self.block = (nn.Sequential(*(RepVGGBlock(out_channels, out_channels)
                                      for _ in range(n - 1)))
                      if n > 1 else None)

    def forward(self, x):
        x = self.conv1(x)
        if self.block is not None:
            x = self.block(x)
        return x


class RepVGGBlock(nn.Layer):
    """RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 padding_mode="zeros",
                 deploy=False,
                 use_se=False):
        super(RepVGGBlock, self).__init__()
        """ Intialization of the class.
        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of
                the input. Default: 1
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            padding_mode (string, optional): Default: 'zeros'
            deploy: Whether to be deploy status or training status. Default: False
            use_se: Whether to use se. Default: False
        """
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias_attr=True,
                padding_mode=padding_mode)

        else:
            self.rbr_identity = (nn.BatchNorm2D(
                num_features=in_channels, epsilon=1e-3, momentum=0.97)
                                 if out_channels == in_channels and stride == 1
                                 else None)
            self.rbr_dense = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups)
            self.rbr_1x1 = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=padding_11,
                groups=groups)

    def forward(self, inputs):
        """Forward process"""
        if hasattr(self, "rbr_reparam"):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(
            self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
                bias3x3 + bias1x1 + biasid)

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2D)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = paddle.to_tensor(kernel_value)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, "rbr_reparam"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2D(
            in_channels=self.rbr_dense.conv.in_channels,
            out_channels=self.rbr_dense.conv.out_channels,
            kernel_size=self.rbr_dense.conv.kernel_size,
            stride=self.rbr_dense.conv.stride,
            padding=self.rbr_dense.conv.padding,
            dilation=self.rbr_dense.conv.dilation,
            groups=self.rbr_dense.conv.groups,
            bias_attr=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("rbr_dense")
        self.__delattr__("rbr_1x1")
        if hasattr(self, "rbr_identity"):
            self.__delattr__("rbr_identity")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")
        self.deploy = True


class SimConv(nn.Layer):
    """Normal Conv with ReLU activation"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 groups=1,
                 bias=False):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias_attr=bias)
        self.bn = nn.BatchNorm2D(out_channels, epsilon=1e-3, momentum=0.97)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SimSPPF(nn.Layer):
    """Simplified SPPF with ReLU activation"""

    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        c_ = in_channels // 2  # hidden channels
        self.cv1 = SimConv(in_channels, c_, 1, 1)
        self.cv2 = SimConv(c_ * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2D(
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(paddle.concat([x, y1, y2, self.m(y2)], 1))


class Transpose(nn.Layer):
    '''Normal Transpose, default for upsampling'''

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.upsample_transpose = nn.Conv2DTranspose(
            in_channels=in_channels,
            out_channels=out_channels,
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
    __shared__ = ['width_mult', 'depth_mult', 'trt']

    def __init__(self,
                 width_mult=1.0,
                 depth_mult=1.0,
                 num_repeats=[1, 6, 12, 18, 6],
                 channels_list=[64, 128, 256, 512, 1024],
                 return_idx=[2, 3, 4],
                 depthwise=False,
                 trt=False):
        super().__init__()
        in_channels = 3
        num_repeats = [(max(round(i * depth_mult), 1) if i > 1 else i)
                       for i in (num_repeats)]
        channels_list = [
            make_divisible(i * width_mult, 8) for i in (channels_list)
        ]

        self._out_channels = channels_list[1:]
        self._out_strides = [4, 8, 16, 32]
        self.return_idx = return_idx

        self.stem = RepVGGBlock(
            in_channels=in_channels,
            out_channels=channels_list[0],
            kernel_size=3,
            stride=2)

        self.ERBlock_2 = nn.Sequential(
            RepVGGBlock(
                in_channels=channels_list[0],
                out_channels=channels_list[1],
                kernel_size=3,
                stride=2),
            RepBlock(
                in_channels=channels_list[1],
                out_channels=channels_list[1],
                n=num_repeats[1]))

        self.ERBlock_3 = nn.Sequential(
            RepVGGBlock(
                in_channels=channels_list[1],
                out_channels=channels_list[2],
                kernel_size=3,
                stride=2),
            RepBlock(
                in_channels=channels_list[2],
                out_channels=channels_list[2],
                n=num_repeats[2]))

        self.ERBlock_4 = nn.Sequential(
            RepVGGBlock(
                in_channels=channels_list[2],
                out_channels=channels_list[3],
                kernel_size=3,
                stride=2),
            RepBlock(
                in_channels=channels_list[3],
                out_channels=channels_list[3],
                n=num_repeats[3]))

        self.ERBlock_5 = nn.Sequential(
            RepVGGBlock(
                in_channels=channels_list[3],
                out_channels=channels_list[4],
                kernel_size=3,
                stride=2),
            RepBlock(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                n=num_repeats[4]),
            SimSPPF(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                kernel_size=5))

    def forward(self, inputs):
        x = inputs['image']
        x = self.stem(x)
        outputs = []
        x = self.ERBlock_2(x)
        x = self.ERBlock_3(x)
        outputs.append(x)
        x = self.ERBlock_4(x)
        outputs.append(x)
        x = self.ERBlock_5(x)
        outputs.append(x)
        return outputs

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=self._out_channels[i], stride=self._out_strides[i])
            for i in self.return_idx
        ]
