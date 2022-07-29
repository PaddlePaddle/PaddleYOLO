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
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from ppdet.core.workspace import register, serializable
from ppdet.modeling.initializer import conv_init_, normal_
from paddle.nn.initializer import Constant
from ..shape_spec import ShapeSpec
from IPython import embed

__all__ = [
    'CSPDarkNet', 'BaseConv', 'DWConv', 'BottleNeck', 'SPPLayer', 'SPPFLayer',
    'ELANNet', 'ELANLayer', 'MPConvLayer', 'SPPCSPC', 'SPPELAN', 'RepConv',
    'MP', 'ImplicitA', 'ImplicitM'
]


def get_activation(name="silu", inplace=True):
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
        self.act = get_activation(act)
        self._init_weights()

    def _init_weights(self):
        conv_init_(self.conv)

    def forward(self, x):
        # use 'x * F.sigmoid(x)' replace 'silu'
        x = self.bn(self.conv(x))
        y = self.act(x)
        # y = x * F.sigmoid(x)
        return y


class DWConv(nn.Layer):
    """Depthwise Conv"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride=1,
                 bias=False,
                 act="silu"):
        super(DWConv, self).__init__()
        self.dw_conv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            bias=bias,
            act=act)
        self.pw_conv = BaseConv(
            in_channels,
            out_channels,
            ksize=1,
            stride=1,
            groups=1,
            bias=bias,
            act=act)

    def forward(self, x):
        return self.pw_conv(self.dw_conv(x))


class Focus(nn.Layer):
    """Focus width and height information into channel space, used in YOLOX."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize=3,
                 stride=1,
                 bias=False,
                 act="silu"):
        super(Focus, self).__init__()
        self.conv = BaseConv(
            in_channels * 4,
            out_channels,
            ksize=ksize,
            stride=stride,
            bias=bias,
            act=act)

    def forward(self, inputs):
        # inputs [bs, C, H, W] -> outputs [bs, 4C, W/2, H/2]
        top_left = inputs[:, :, 0::2, 0::2]
        top_right = inputs[:, :, 0::2, 1::2]
        bottom_left = inputs[:, :, 1::2, 0::2]
        bottom_right = inputs[:, :, 1::2, 1::2]
        outputs = paddle.concat(
            [top_left, bottom_left, top_right, bottom_right], 1)
        return self.conv(outputs)


class BottleNeck(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 shortcut=True,
                 expansion=0.5,
                 depthwise=False,
                 bias=False,
                 act="silu"):
        super(BottleNeck, self).__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        self.conv2 = Conv(
            hidden_channels,
            out_channels,
            ksize=3,
            stride=1,
            bias=bias,
            act=act)
        self.add_shortcut = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.add_shortcut:
            y = y + x
        return y


class SPPLayer(nn.Layer):
    """Spatial Pyramid Pooling (SPP) layer used in YOLOv3-SPP and YOLOX"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes=(5, 9, 13),
                 bias=False,
                 act="silu"):
        super(SPPLayer, self).__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        self.maxpoolings = nn.LayerList([
            nn.MaxPool2D(
                kernel_size=ks, stride=1, padding=ks // 2)
            for ks in kernel_sizes
        ])
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(
            conv2_channels, out_channels, ksize=1, stride=1, bias=bias, act=act)

    def forward(self, x):
        x = self.conv1(x)
        x = paddle.concat([x] + [mp(x) for mp in self.maxpoolings], axis=1)
        x = self.conv2(x)
        return x


class SPPFLayer(nn.Layer):
    """ Spatial Pyramid Pooling - Fast (SPPF) layer used in YOLOv5 by Glenn Jocher,
        equivalent to SPP(k=(5, 9, 13))
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize=5,
                 bias=False,
                 act='silu'):
        super(SPPFLayer, self).__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        self.maxpooling = nn.MaxPool2D(
            kernel_size=ksize, stride=1, padding=ksize // 2)
        conv2_channels = hidden_channels * 4
        self.conv2 = BaseConv(
            conv2_channels, out_channels, ksize=1, stride=1, bias=bias, act=act)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.maxpooling(x)
        y2 = self.maxpooling(y1)
        y3 = self.maxpooling(y2)
        concats = paddle.concat([x, y1, y2, y3], axis=1)
        out = self.conv2(concats)
        return out


class CSPLayer(nn.Layer):
    """CSP (Cross Stage Partial) layer with 3 convs, named C3 in YOLOv5"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=1,
                 shortcut=True,
                 expansion=0.5,
                 depthwise=False,
                 bias=False,
                 act="silu"):
        super(CSPLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        self.conv2 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(* [
            BottleNeck(
                hidden_channels,
                hidden_channels,
                shortcut=shortcut,
                expansion=1.0,
                depthwise=depthwise,
                bias=bias,
                act=act) for _ in range(num_blocks)
        ])
        self.conv3 = BaseConv(
            hidden_channels * 2,
            out_channels,
            ksize=1,
            stride=1,
            bias=bias,
            act=act)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        x = paddle.concat([x_1, x_2], axis=1)
        x = self.conv3(x)
        return x


@register
@serializable
class CSPDarkNet(nn.Layer):
    """
    CSPDarkNet backbone.
    Args:
        arch (str): Architecture of CSPDarkNet, from {P5, P6, X}, default as X,
            and 'X' means used in YOLOX, 'P5/P6' means used in YOLOv5.
        depth_mult (float): Depth multiplier, multiply number of channels in
            each layer, default as 1.0.
        width_mult (float): Width multiplier, multiply number of blocks in
            CSPLayer, default as 1.0.
        depthwise (bool): Whether to use depth-wise conv layer.
        act (str): Activation function type, default as 'silu'.
        return_idx (list): Index of stages whose feature maps are returned.
    """

    __shared__ = ['depth_mult', 'width_mult', 'act', 'trt']

    # in_channels, out_channels, num_blocks, add_shortcut, use_spp(use_sppf)
    # 'X' means setting used in YOLOX, 'P5/P6' means setting used in YOLOv5.
    arch_settings = {
        'X': [[64, 128, 3, True, False], [128, 256, 9, True, False],
              [256, 512, 9, True, False], [512, 1024, 3, False, True]],
        'P5': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 9, True, False], [512, 1024, 3, True, True]],
        'P6': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 9, True, False], [512, 768, 3, True, False],
               [768, 1024, 3, True, True]],
    }

    def __init__(self,
                 arch='X',
                 depth_mult=1.0,
                 width_mult=1.0,
                 depthwise=False,
                 act='silu',
                 trt=False,
                 return_idx=[2, 3, 4]):
        super(CSPDarkNet, self).__init__()
        self.arch = arch
        self.return_idx = return_idx
        Conv = DWConv if depthwise else BaseConv
        arch_setting = self.arch_settings[arch]
        base_channels = int(arch_setting[0][0] * width_mult)

        # Note: differences between the latest YOLOv5 and the original YOLOX
        # 1. self.stem, use SPPF(in YOLOv5) or SPP(in YOLOX)
        # 2. use SPPF(in YOLOv5) or SPP(in YOLOX)
        # 3. put SPPF before(YOLOv5) or SPP after(YOLOX) the last cspdark block's CSPLayer
        # 4. whether SPPF(SPP)'CSPLayer add shortcut, True in YOLOv5, False in YOLOX
        if arch in ['P5', 'P6']:
            # in the latest YOLOv5, use Conv stem, and SPPF (fast, only single spp kernal size)
            self.stem = Conv(
                3, base_channels, ksize=6, stride=2, bias=False, act=act)
            spp_kernal_sizes = 5
        elif arch in ['X']:
            # in the original YOLOX, use Focus stem, and SPP (three spp kernal sizes)
            self.stem = Focus(
                3, base_channels, ksize=3, stride=1, bias=False, act=act)
            spp_kernal_sizes = (5, 9, 13)
        else:
            raise AttributeError("Unsupported arch type: {}".format(arch))

        _out_channels = [base_channels]
        layers_num = 1
        self.csp_dark_blocks = []

        for i, (in_channels, out_channels, num_blocks, shortcut,
                use_spp) in enumerate(arch_setting):
            in_channels = int(in_channels * width_mult)
            out_channels = int(out_channels * width_mult)
            _out_channels.append(out_channels)
            num_blocks = max(round(num_blocks * depth_mult), 1)
            stage = []

            conv_layer = self.add_sublayer(
                'layers{}.stage{}.conv_layer'.format(layers_num, i + 1),
                Conv(
                    in_channels, out_channels, 3, 2, bias=False, act=act))
            stage.append(conv_layer)
            layers_num += 1

            if use_spp and arch in ['X']:
                # in YOLOX use SPPLayer
                spp_layer = self.add_sublayer(
                    'layers{}.stage{}.spp_layer'.format(layers_num, i + 1),
                    SPPLayer(
                        out_channels,
                        out_channels,
                        kernel_sizes=spp_kernal_sizes,
                        bias=False,
                        act=act))
                stage.append(spp_layer)
                layers_num += 1

            csp_layer = self.add_sublayer(
                'layers{}.stage{}.csp_layer'.format(layers_num, i + 1),
                CSPLayer(
                    out_channels,
                    out_channels,
                    num_blocks=num_blocks,
                    shortcut=shortcut,
                    depthwise=depthwise,
                    bias=False,
                    act=act))
            stage.append(csp_layer)
            layers_num += 1

            if use_spp and arch in ['P5', 'P6']:
                # in latest YOLOv5 use SPPFLayer instead of SPPLayer
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

            self.csp_dark_blocks.append(nn.Sequential(*stage))

        self._out_channels = [_out_channels[i] for i in self.return_idx]
        self.strides = [[2, 4, 8, 16, 32, 64][i] for i in self.return_idx]

    def forward(self, inputs):
        x = inputs['image']
        outputs = []
        x = self.stem(x)
        for i, layer in enumerate(self.csp_dark_blocks):
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


##### yolov7 ####


class ELANLayer(nn.Layer):
    """ELAN layer used in YOLOv7"""

    def __init__(self,
                 in_channels,
                 mid_channels1,
                 mid_channels2,
                 out_channels,
                 num_blocks=4,
                 concat_list=[-1, -3, -5, -6],
                 depthwise=False,
                 bias=False,
                 act="silu"):
        super(ELANLayer, self).__init__()
        self.num_blocks = num_blocks
        self.concat_list = concat_list

        self.conv1 = BaseConv(
            in_channels, mid_channels1, ksize=1, stride=1, bias=bias, act=act)
        self.conv2 = BaseConv(
            in_channels, mid_channels1, ksize=1, stride=1, bias=bias, act=act)

        self.bottlenecks = nn.Sequential(* [
            BaseConv(
                mid_channels1 if i == 0 else mid_channels2,
                mid_channels2,
                ksize=3,
                stride=1,
                bias=bias,
                act=act) for i in range(num_blocks)
        ])

        concat_chs = mid_channels1 * 2 + mid_channels2 * (len(concat_list) - 2)
        self.conv3 = BaseConv(
            int(concat_chs),
            out_channels,
            ksize=1,
            stride=1,
            bias=bias,
            act=act)

    def forward(self, x):
        outs = []
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        outs.append(x_1)
        outs.append(x_2)
        idx = [i + self.num_blocks for i in self.concat_list[:-2]]
        for i in range(self.num_blocks):
            x_2 = self.bottlenecks[i](x_2)
            if i in idx:
                outs.append(x_2)
        outs = outs[::-1]  # [-1, -3]
        x_all = paddle.concat(outs, axis=1)
        y = self.conv3(x_all)
        return y


class MPConvLayer(nn.Layer):
    """MPConvLayer used in YOLOv7"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=0.5,
                 depthwise=False,
                 bias=False,
                 act="silu"):
        super(MPConvLayer, self).__init__()
        mid_channels = int(out_channels * expansion)
        self.maxpool = nn.MaxPool2D(kernel_size=2, stride=2)
        self.conv1 = BaseConv(
            in_channels, mid_channels, ksize=1, stride=1, bias=bias, act=act)

        self.conv2 = BaseConv(
            in_channels, mid_channels, ksize=1, stride=1, bias=bias, act=act)
        self.conv3 = BaseConv(
            mid_channels, mid_channels, ksize=3, stride=2, bias=bias, act=act)

    def forward(self, x):
        x_1 = self.conv1(self.maxpool(x))
        x_2 = self.conv3(self.conv2(x))
        x = paddle.concat([x_2, x_1], axis=1)
        return x


class MP(nn.Layer):
    def __init__(self, kernel_size=2, stride=2):
        super(MP, self).__init__()
        self.mp = nn.MaxPool2D(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return self.mp(x)


class DownC(nn.Layer):
    def __init__(self, c1, c2, k=2, act='silu'):
        super(DownC, self).__init__()
        c_ = int(c1)  # hidden channels
        self.mp = nn.MaxPool2D(kernel_size=k, stride=k)
        self.cv1 = BaseConv(c1, c_, 1, 1, act=act)
        self.cv2 = BaseConv(c_, c2 // 2, 3, k, act=act)
        self.cv3 = BaseConv(c1, c2 // 2, 1, 1, act=act)

    def forward(self, x):
        x_2 = self.cv2(self.cv1(x))
        x_3 = self.cv3(self.mp(x))
        return paddle.concat([x_2, x_3], 1)


class SPPCSPC(nn.Layer):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, g=1, e=0.5, k=(5, 9, 13), act='silu'):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = BaseConv(c1, c_, 1, 1, act=act)
        self.cv2 = BaseConv(c1, c_, 1, 1, act=act)
        self.cv3 = BaseConv(c_, c_, 3, 1, act=act)
        self.cv4 = BaseConv(c_, c_, 1, 1, act=act)
        self.maxpoolings = nn.LayerList(
            [nn.MaxPool2D(
                kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = BaseConv(4 * c_, c_, 1, 1, act=act)
        self.cv6 = BaseConv(c_, c_, 3, 1, act=act)
        self.cv7 = BaseConv(2 * c_, c2, 1, 1, act=act)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(
            self.cv5(
                paddle.concat([x1] + [mp(x1) for mp in self.maxpoolings], 1)))
        y2 = self.cv2(x)
        return self.cv7(paddle.concat([y1, y2], axis=1))


class SPPELAN(nn.Layer):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, g=1, e=0.5, k=(5, 9, 13), act='silu'):
        super(SPPELAN, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = BaseConv(c1, c_, 1, 1, act=act)
        self.cv2 = BaseConv(c1, c_, 1, 1, act=act)
        self.maxpoolings = nn.LayerList(
            [nn.MaxPool2D(
                kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv3 = BaseConv(4 * c_, c_, 1, 1, act=act)
        self.cv4 = BaseConv(2 * c_, c2, 1, 1, act=act)

    def forward(self, x):
        x_1 = self.cv1(x)
        x_2 = self.cv2(x)
        x_cats = [x_2] + [mp(x_2) for mp in self.maxpoolings]
        y_cats = self.cv3(paddle.concat(x_cats[::-1], 1))
        y = paddle.concat([y_cats, x_1], 1)
        return self.cv4(y)


class ImplicitA(nn.Layer):
    def __init__(self, channel, mean=0., std=.02):
        super(ImplicitA, self).__init__()
        self.implicit = self.create_parameter(
            shape=([1, channel, 1, 1]),
            attr=ParamAttr(initializer=Constant(0.)))
        normal_(self.implicit, mean=mean, std=std)

    def forward(self, x):
        return self.implicit + x


class ImplicitM(nn.Layer):
    def __init__(self, channel, mean=0., std=.02):
        super(ImplicitM, self).__init__()
        self.implicit = self.create_parameter(
            shape=([1, channel, 1, 1]),
            attr=ParamAttr(initializer=Constant(1.)))
        normal_(self.implicit, mean=mean, std=std)

    def forward(self, x):
        return self.implicit * x


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class RepConv(nn.Layer):
    # RepVGG, see https://arxiv.org/abs/2101.03697
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True, deploy=False):
        super(RepConv, self).__init__()
        self.deploy = deploy
        self.groups = g
        self.in_channels = c1
        self.out_channels = c2
        assert k == 3
        assert autopad(k, p) == 1
        padding_11 = autopad(k, p) - k // 2

        #if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.act = get_activation(act)

        if deploy:
            self.rbr_reparam = nn.Conv2D(
                c1, c2, k, s, autopad(k, p), groups=g, bias_attr=True)
        else:
            self.rbr_identity = (nn.BatchNorm2D(c1)
                                 if c2 == c1 and s == 1 else None)
            self.rbr_dense = nn.Sequential(* [
                nn.Conv2D(
                    c1, c2, k, s, autopad(k, p), groups=g, bias_attr=False),
                nn.BatchNorm2D(c2),
            ])
            self.rbr_1x1 = nn.Sequential(* [
                nn.Conv2D(
                    c1, c2, 1, s, padding_11, groups=g, bias_attr=False),
                nn.BatchNorm2D(c2),
            ])

    def forward(self, inputs):
        # [8, 128, 80, 80] [8, 256, 40, 40] [8, 512, 20, 20]
        if hasattr(self, "rbr_reparam"):
            x = self.rbr_reparam(inputs)
            y = self.act(x)
            # y = x * F.sigmoid(x)
            return y

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        x = self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out
        y = self.act(x)
        # y = x * F.sigmoid(x)
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
                self.id_tensor = kernel_value  #torch.from_numpy(kernel_value).to(branch.weight.device)
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
        if hasattr(self, 'rbr_reparam'):
            return
        print(f"RepConv_OREPA.switch_to_deploy")
        kernel, bias = self.get_equivalent_kernel_bias()
        # self.rbr_reparam = nn.Conv2D(
        #     self.rbr_dense.in_channels,
        #     self.rbr_dense.out_channels,
        #     self.rbr_dense.kernel_size,
        #     self.rbr_dense.stride,
        #     self.rbr_dense.padding,
        #     dilation=self.rbr_dense.dilation,
        #     groups=self.rbr_dense.groups,
        #     bias_attr=True)
        if not hasattr(self, 'rbr_reparam'):
            self.conv = nn.Conv2D(
                self.in_channels,
                self.out_channels,
                3,
                1,
                1,
                groups=self.groups,
                bias_attr=True)
        return (kernel.numpy(), bias.numpy())

        # self.rbr_reparam.weight.set_value(kernel)
        # self.rbr_reparam.bias.set_value(bias)
        # # for para in self.parameters():
        # #     para.detach_()
        # self.__delattr__('rbr_dense')
        # self.__delattr__('rbr_1x1')
        # if hasattr(self, 'rbr_identity'):
        #     self.__delattr__('rbr_identity') 
        # return (kernel.numpy(), bias.numpy())

    def fuse_conv_bn(self, conv, bn):
        std = (bn.running_var + bn.eps).sqrt()
        bias = bn.bias - bn.running_mean * bn.weight / std
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        weights = conv.weight * t
        bn = nn.Identity()
        conv = nn.Conv2D(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
            padding_mode=conv.padding_mode)
        conv.weight.set_value(weights)
        conv.bias.set_value(bias)
        return conv

    def fuse_repvgg_block(self):
        if self.deploy:
            return
        print(f"RepConv.fuse_repvgg_block")
        self.rbr_dense = self.fuse_conv_bn(self.rbr_dense[0], self.rbr_dense[1])
        self.rbr_1x1 = self.fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
        rbr_1x1_bias = self.rbr_1x1.bias
        weight_1x1_expanded = nn.functional.pad(self.rbr_1x1.weight,
                                                [1, 1, 1, 1])

        # Fuse self.rbr_identity
        if isinstance(
                self.rbr_identity, nn.BatchNorm2D
        ):  #or isinstance(self.rbr_identity, nn.modules.batchnorm.SyncBatchNorm)):
            # print(f"fuse: rbr_identity == BatchNorm2D or SyncBatchNorm")
            identity_conv_1x1 = nn.Conv2D(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=self.groups,
                bias=False)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.to(
                self.rbr_1x1.weight.data.device)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.squeeze(
            ).squeeze()
            # print(f" identity_conv_1x1.weight = {identity_conv_1x1.weight.shape}")
            identity_conv_1x1.weight.data.fill_(0.0)
            identity_conv_1x1.weight.data.fill_diagonal_(1.0)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.unsqueeze(
                2).unsqueeze(3)
            # print(f" identity_conv_1x1.weight = {identity_conv_1x1.weight.shape}")

            identity_conv_1x1 = self.fuse_conv_bn(identity_conv_1x1,
                                                  self.rbr_identity)
            bias_identity_expanded = identity_conv_1x1.bias
            weight_identity_expanded = nn.functional.pad(
                identity_conv_1x1.weight, [1, 1, 1, 1])
        else:
            # print(f"fuse: rbr_identity != BatchNorm2d, rbr_identity = {self.rbr_identity}")
            bias_identity_expanded = nn.Parameter(
                paddle.zeros_like(rbr_1x1_bias))
            weight_identity_expanded = nn.Parameter(
                paddle.zeros_like(weight_1x1_expanded))

        self.rbr_dense.weight = nn.Parameter(self.rbr_dense.weight +
                                             weight_1x1_expanded +
                                             weight_identity_expanded)
        self.rbr_dense.bias = nn.Parameter(self.rbr_dense.bias + rbr_1x1_bias +
                                           bias_identity_expanded)

        self.rbr_reparam = self.rbr_dense
        self.deploy = True

        if self.rbr_identity is not None:
            del self.rbr_identity
            self.rbr_identity = None

        if self.rbr_1x1 is not None:
            del self.rbr_1x1
            self.rbr_1x1 = None

        if self.rbr_dense is not None:
            del self.rbr_dense
            self.rbr_dense = None


@register
@serializable
class ELANNet(nn.Layer):
    """
    ELANNet, YOLOv7's backbone.
    Args:
        arch (str): Architecture of ELANNet, from {tiny, L, X, W6, E6, D6, E6E}, default as 'L',
        depth_mult (float): Depth multiplier, multiply number of channels in
            each layer, default as 1.0.
        width_mult (float): Width multiplier, multiply number of blocks in
            CSPLayer, default as 1.0.
        depthwise (bool): Whether to use depth-wise conv layer.
        act (str): Activation function type, default as 'silu'.
        trt (bool): Whether use trt infer.
        return_idx (list): Index of stages whose feature maps are returned.
    """
    __shared__ = ['depth_mult', 'width_mult', 'act', 'trt']

    # in_channels, out_channels of 1 stem + 4 stages
    ch_settings = {
        'tiny': [[32, 64], [64, 64], [64, 128], [128, 256], [256, 512]],
        'L': [[32, 64], [64, 256], [256, 512], [512, 1024], [1024, 1024]],
        'X': [[40, 80], [80, 320], [320, 640], [640, 1280], [1280, 1280]],
        'W6':
        [[64, 64], [64, 128], [128, 256], [256, 512], [512, 768], [768, 1024]],
        'E6':
        [[80, 80], [80, 160], [160, 320], [320, 640], [640, 960], [960, 1280]],
        'D6': [[96, 96], [96, 192], [192, 384], [384, 768], [768, 1152],
               [1152, 1536]],
        'E6E':
        [[80, 80], [80, 160], [160, 320], [320, 640], [640, 960], [960, 1280]],
    }
    # mid_ch1, mid_ch2 of 4 stages' ELANLayer
    mid_ch_settings = {
        'tiny': [[32, 32], [64, 64], [128, 128], [256, 256]],
        'L': [[64, 64], [128, 128], [256, 256], [256, 256]],
        'X': [[64, 64], [128, 128], [256, 256], [256, 256]],
        'W6': [[64, 64], [128, 128], [256, 256], [384, 384], [512, 512]],
        'E6': [[64, 64], [128, 128], [256, 256], [384, 384], [512, 512]],
        'D6': [[64, 64], [128, 128], [256, 256], [384, 384], [512, 512]],
        'E6E': [[64, 64], [128, 128], [256, 256], [384, 384], [512, 512]],
    }
    # concat_list of 4 stages
    concat_list_settings = {
        'tiny': [-1, -2, -3, -4],
        'L': [-1, -3, -5, -6],
        'X': [-1, -3, -5, -7, -8],
        'W6': [-1, -3, -5, -6],
        'E6': [-1, -3, -5, -7, -8],
        'D6': [-1, -3, -5, -7, -9, -10],
        'E6E': [-1, -3, -5, -7, -8],
    }
    num_blocks = {
        'tiny': 2,
        'L': 4,
        'X': 6,
        'W6': 4,
        'E6': 6,
        'D6': 8,
        'E6E': 6
    }

    def __init__(self,
                 arch='L',
                 depth_mult=1.0,
                 width_mult=1.0,
                 depthwise=False,
                 act='silu',
                 trt=False,
                 return_idx=[2, 3, 4]):
        super(ELANNet, self).__init__()
        self.arch = arch
        self.return_idx = return_idx
        Conv = DWConv if depthwise else BaseConv

        ch_settings = self.ch_settings[arch]  # * width_mult
        mid_ch_settings = self.mid_ch_settings[arch]
        concat_list_settings = self.concat_list_settings[arch]
        num_blocks = self.num_blocks[arch]

        layers_num = 0
        ch_1 = ch_settings[0][0]
        ch_2 = ch_settings[0][0] * 2
        ch_out = ch_settings[0][-1]
        if self.arch in ['L', 'X']:
            self.stem = nn.Sequential(* [
                Conv(
                    3, ch_1, 3, 1, bias=False, act=act),
                Conv(
                    ch_1, ch_2, 3, 2, bias=False, act=act),
                Conv(
                    ch_2, ch_out, 3, 1, bias=False, act=act),
            ])
            layers_num = 3
        elif self.arch in ['tiny']:
            self.stem = nn.Sequential(* [
                Conv(
                    3, ch_1, 3, 2, bias=False, act=act),
                Conv(
                    ch_1, ch_out, 3, 2, bias=False, act=act),
            ])
            layers_num = 2
        elif self.arch in ['W6', 'E6', 'D6', 'E6E']:
            self.stem = Focus(3, ch_out, 3, 1, bias=False, act=act)
            layers_num = 2
        else:
            raise AttributeError("Unsupported arch type: {}".format(self.arch))

        self._out_channels = [chs[-1] for chs in ch_settings]
        self._out_channels[
            -1] //= 2  # for SPPCSPC(L,X,W6,E6,D6,E6E) or SPPELAN(tiny)
        self._out_channels = [self._out_channels[i] for i in self.return_idx]
        self.strides = [[2, 4, 8, 16, 32, 64][i] for i in self.return_idx]

        self.blocks = []
        for i, (in_ch, out_ch) in enumerate(ch_settings[1:]):
            stage = []

            # 1.Downsample methods: Conv, DownC, MPConvLayer, MP, None
            if i == 0:
                if self.arch in ['L', 'X', 'W6']:
                    # Conv
                    _out_ch = out_ch if self.arch == 'W6' else out_ch // 2
                    conv_layer = self.add_sublayer(
                        'layers{}.stage{}.conv_layer'.format(layers_num, i + 1),
                        Conv(
                            in_ch, _out_ch, 3, 2, bias=False, act=act))
                    stage.append(conv_layer)
                    layers_num += 1
                elif self.arch in ['E6', 'D6', 'E6E']:
                    # DownC
                    downc_layer = self.add_sublayer(
                        'layers{}.stage{}.downc_layer'.format(layers_num,
                                                              i + 1),
                        DownC(
                            in_ch, out_ch, 2, act=act))
                    stage.append(downc_layer)
                    layers_num += 1
                elif self.arch in ['tiny']:
                    # None
                    pass
                else:
                    raise AttributeError("Unsupported arch type: {}".format(
                        self.arch))
            else:
                if self.arch in ['L', 'X']:
                    # MPConvLayer
                    # Note: out channels of MPConvLayer is int(in_ch * 0.5 * 2)
                    # no relationship with out_ch when used in backbone
                    conv_res_layer = self.add_sublayer(
                        'layers{}.stage{}.mpconv_layer'.format(layers_num,
                                                               i + 1),
                        MPConvLayer(
                            in_ch, in_ch, 0.5, depthwise, bias=False, act=act))
                    stage.append(conv_res_layer)
                    layers_num += 5  # 1 maxpool + 3 convs + 1 concat
                elif self.arch in ['tiny']:
                    # MP
                    mp_layer = self.add_sublayer(
                        'layers{}.stage{}.mp_layer'.format(layers_num, i + 1),
                        MP(kernel_size=2, stride=2))
                    stage.append(mp_layer)
                    layers_num += 1
                elif self.arch in ['W6']:
                    # Conv
                    conv_layer = self.add_sublayer(
                        'layers{}.stage{}.conv_layer'.format(layers_num, i + 1),
                        Conv(
                            in_ch, out_ch, 3, 2, bias=False, act=act))
                    stage.append(conv_layer)
                    layers_num += 1
                elif self.arch in ['E6', 'D6', 'E6E']:
                    # DownC
                    downc_layer = self.add_sublayer(
                        'layers{}.stage{}.downc_layer'.format(layers_num,
                                                              i + 1),
                        DownC(
                            in_ch, out_ch, 2, act=act))
                    stage.append(downc_layer)
                    layers_num += 1
                else:
                    raise AttributeError("Unsupported arch type: {}".format(
                        self.arch))

            # 2.ELANLayer Block: like CSPLayer(C3) in YOLOv5/YOLOX
            elan_in_ch = in_ch
            if i == 0 and self.arch in ['L', 'X']:
                elan_in_ch = in_ch * 2
            if self.arch in ['W6', 'E6', 'D6', 'E6E']:
                elan_in_ch = out_ch
            elan_layer = self.add_sublayer(
                'layers{}.stage{}.elan_layer'.format(layers_num, i + 1),
                ELANLayer(
                    elan_in_ch,
                    mid_ch_settings[i][0],
                    mid_ch_settings[i][1],
                    out_ch,
                    num_blocks=num_blocks,
                    concat_list=concat_list_settings,
                    depthwise=depthwise,
                    bias=False,
                    act=act))
            stage.append(elan_layer)
            layers_num += int(2 + num_blocks + 2)
            # conv1 + conv2 + bottleneck + concat + conv3

            # 3.SPP(Spatial Pyramid Pooling) methods: SPPCSPC, SPPELAN
            if i == len(ch_settings[1:]) - 1:
                if self.arch in ['L', 'X', 'W6', 'E6', 'D6', 'E6E']:
                    sppcspc_layer = self.add_sublayer(
                        'layers{}.stage{}.sppcspc_layer'.format(layers_num,
                                                                i + 1),
                        SPPCSPC(
                            out_ch, out_ch // 2, k=(5, 9, 13), act=act))
                    stage.append(sppcspc_layer)
                    layers_num += 1
                elif self.arch in ['tiny']:
                    sppelan_layer = self.add_sublayer(
                        'layers{}.stage{}.sppelan_layer'.format(layers_num,
                                                                i + 1),
                        SPPELAN(
                            out_ch, out_ch // 2, k=(5, 9, 13), act=act))
                    stage.append(sppelan_layer)
                    layers_num += 9
                else:
                    raise AttributeError("Unsupported arch type: {}".format(
                        self.arch))

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
