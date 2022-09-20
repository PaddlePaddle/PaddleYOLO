from __future__ import division

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from paddle.nn import Conv2D, MaxPool2D
from ppdet.core.workspace import register, serializable
from ..shape_spec import ShapeSpec
from ppdet.modeling.ops import get_act_fn
import warnings
__all__ = ['RepVGGYOLOX']

VGG_cfg = {16: [2, 2, 3, 3, 3], 19: [2, 2, 4, 4, 4]}
RepVGGX=[1, 2, 4, 6, 2, 4, 4, 4, 4]


class ConvBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 groups,
                 pool_size=2,
                 pool_stride=2,
                 pool_padding=0,
                 name=None,
                 add_ppf=False):
        super(ConvBlock, self).__init__()

        self.groups = groups
        conv_out_list = []
        conv_out_list.append(RepVggBlock(
            ch_in=in_channels,
            ch_out=out_channels,stride=2))
        self.add_ppf = add_ppf
        for i in range(groups):
            conv_out = self.add_sublayer(
                'conv{}'.format(i),
                RepVggBlock(
                    ch_in=out_channels,
                    ch_out=out_channels,
                    padding=1))
            conv_out_list.append(conv_out)

        if self.add_ppf: 
            # conv_out_list.append(MT_SPPF(out_channels, out_channels, kernel_size=5))
            self.mt=MT_SPPF(out_channels, out_channels, kernel_size=5)
        self.conv_out_list=nn.Sequential(*conv_out_list)

    def forward(self, inputs):
        out=self.conv_out_list(inputs)
        if self.add_ppf: 
           out =  self.mt(out)
        # if self.add_ppf:
        # #     try:
        #     if len(self.conv_out_list)<2:
        #             raise IndexError 
        #         out = self.conv_out_list[-1](out)#看看正不正确
        #     except IndexError as e:
        #         print("引发异常：",repr(e))

            
       
        return out


class ExtraBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 padding,
                 stride,
                 kernel_size,
                 name=None):
        super(ExtraBlock, self).__init__()

        self.conv0 = Conv2D(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            padding=0)
        self.conv1 = Conv2D(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)

    def forward(self, inputs):
        out = self.conv0(inputs)
        out = F.relu(out)
        out = self.conv1(out)
        out = F.relu(out)
        return out


class L2NormScale(nn.Layer):
    def __init__(self, num_channels, scale=1.0):
        super(L2NormScale, self).__init__()
        self.scale = self.create_parameter(
            attr=ParamAttr(initializer=paddle.nn.initializer.Constant(scale)),
            shape=[num_channels])

    def forward(self, inputs):
        out = F.normalize(inputs, axis=1, epsilon=1e-10)
        # out = self.scale.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(
        #     out) * out
        out = self.scale.unsqueeze(0).unsqueeze(2).unsqueeze(3) * out
        return out


@register
@serializable
class RepVGGYOLOX(nn.Layer):
    def __init__(self,
                 depth=16,
                 normalizations=[20., -1, -1, -1, -1, -1],
                 ):
        super(RepVGGYOLOX, self).__init__()

        assert depth in [16, 19], \
                "depth as 16/19 supported currently, but got {}".format(depth)
        self.depth = depth
        self.groups = RepVGGX
        self.normalizations = normalizations

        self._out_channels = []
        # self.stage0 = RepVGGBlock(
        #     in_channels=in_channels,
        #     out_channels=channels_list[0],pool_size=2,
        #     ksize=3,
        #     stride=2)
        self.conv_block_0=RepVggBlock(3,32,stride=2)
        self.conv_block_1 = ConvBlock(
            32, 64,self.groups[1], 2, 2,0, name="conv1_")
        self.conv_block_2 = ConvBlock(
            64, 128, self.groups[2], 2, 2, 0, name="conv2_")
        self.conv_block_3 = ConvBlock(
            128, 256, self.groups[3], 2, 2, 0, name="conv3_")
        self.conv_block_4 = ConvBlock(
            256, 512, self.groups[4], 2, 2, 0, name="conv4_",add_ppf=True)
        # self.conv_block_4 = ConvBlock(
        #     512, 512, self.groups[4], 3, 1, 1, name="conv5_")
        # self._out_channels.append(512)

    def forward(self, inputs):
        outputs = []

        conv = self.conv_block_0(inputs['image'])#inputs['image'].shape [8, 3, 480, 480]   
        conv= self.conv_block_1(conv)#[8, 32, 240, 240]
        conv = self.conv_block_2(conv)#[8, 64, 120, 120]
        outputs.append(conv)
        conv = self.conv_block_3(conv)#[8, 128, 60, 60]
        outputs.append(conv)

        out = self.conv_block_4(conv)#[8, 256, 30, 30]
        outputs.append(out)#

        return outputs


    @property
    def out_shape(self):
        return [ShapeSpec(channels=c) for c in self._out_channels]



class RepVggBlock(nn.Layer):#包含了激活函数relu
    def __init__(self, ch_in, ch_out, act='relu', alpha=False, padding=1,stride=1):
        super(RepVggBlock, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvBNLayer(
            ch_in, ch_out, 3, stride=stride, padding=padding, act=None)
        self.conv2 = ConvBNLayer(
            ch_in, ch_out, 1, stride=stride, padding=0, act=None)# padding_11 = padding - ksize // 2
        self.act = get_act_fn(act) if act is None or isinstance(act, (
            str, dict)) else act
        if alpha:
            self.alpha = self.create_parameter(
                shape=[1],
                attr=ParamAttr(initializer=Constant(value=1.)),
                dtype="float32")
        else:
            self.alpha = None

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            if self.alpha:
                y = self.conv1(x) + self.alpha * self.conv2(x)
            else:
                y = self.conv1(x) + self.conv2(x)
        y = self.act(y)
        return y

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2D(
                in_channels=self.ch_in,
                out_channels=self.ch_out,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=1)
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.set_value(kernel)
        self.conv.bias.set_value(bias)
        self.__delattr__('conv1')
        self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        if self.alpha:
            return kernel3x3 + self.alpha * self._pad_1x1_to_3x3_tensor(
                kernel1x1), bias3x3 + self.alpha * bias1x1
        else:
            return kernel3x3 + self._pad_1x1_to_3x3_tensor(
                kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.bn._mean
        running_var = branch.bn._variance
        gamma = branch.bn.weight
        beta = branch.bn.bias
        eps = branch.bn._epsilon
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape((-1, 1, 1, 1))
        return kernel * t, beta - running_mean * gamma / std


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size=3,
                 stride=1,
                 groups=1,
                 padding=0,
                 act=None):
        super(ConvBNLayer, self).__init__()

        self.conv = nn.Conv2D(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias_attr=False)

        self.bn = nn.BatchNorm2D(
            ch_out,
            weight_attr=ParamAttr(regularizer=L2Decay(0.)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.)))
        self.act = get_act_fn(act) if act is None or isinstance(act, (
            str, dict)) else act

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x





class MT_SPPF(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        c_ = in_channels // 2  # hidden channels
        self.cv1 = ConvBNAct(in_channels, c_, 1, 1)
        self.cv2 = ConvBNAct(c_ * 4, out_channels, 1, 1)
        self.maxpool = nn.MaxPool2D(
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.maxpool(x)
            y2 = self.maxpool(y1)
            return self.cv2(paddle.concat([x, y1, y2, self.maxpool(y2)], 1))


class ConvBNAct(nn.Layer):
    '''Normal Conv with SiLU activation'''

    def __init__(self,
                 in_channels,
                 out_channels, 
                 kernel_size,
                 stride,
                 groups=1,
                 bias=False,
                 act='relu'):
        super().__init__()
        padding = kernel_size // 2
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.conv = nn.Conv2D(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                groups=1)
        self.bn = nn.BatchNorm2D(out_channels)

        if act == 'relu':
            self.act = nn.ReLU()
        if act == 'silu':
            self.act = nn.Silu()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))