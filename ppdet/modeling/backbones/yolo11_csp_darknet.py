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

import math
import paddle
from paddle import nn
from ppdet.core.workspace import register, serializable
from .csp_darknet import CSPLayer, SPPFLayer
from ..shape_spec import ShapeSpec

__all__ = [
    "YOLO11CSPDarkNet",
    "Conv",
    "BottleNeck",
    "C2f",
    "C3k2",
    "C3k",
    "Attention",
    "PSABlock",
    "C2PSA",
]


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """
    Calculate padding values to maintain output dimensions with different kernel sizes and dilations.
    
    Args:
        k (int or list): Kernel size.
        p (int or list, optional): Padding value. If None, calculated automatically.
        d (int, optional): Dilation factor. Default is 1.
        
    Returns:
        int or list: Appropriate padding value(s) to maintain output dimensions.
    """
    if d > 1:
        k = (
            d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
        )  # calculate effective kernel size with dilation
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # calculate padding to maintain spatial dimensions
    return p


class Conv(nn.Layer):
    """
    Standard convolution module with batch normalization and activation.
    
    This module combines Conv2D, BatchNorm2D, and activation in a single layer,
    which is a common pattern in modern neural networks.
    """

    default_act = nn.Silu()  # default activation function (Sigmoid Linear Unit)

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """
        Initialize Conv layer with customizable parameters.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size. Default is 1.
            s (int): Stride. Default is 1.
            p (int, optional): Padding. If None, calculated using autopad.
            g (int): Number of groups for grouped convolution. Default is 1.
            d (int): Dilation factor. Default is 1.
            act (bool or nn.Layer): Activation type. If True, uses default_act. Default is True.
        """
        super().__init__()
        self.conv = nn.Conv2D(
            c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias_attr=False
        )
        self.bn = nn.BatchNorm2D(c2)
        self.act = (
            self.default_act
            if act is True
            else act
            if isinstance(act, nn.Layer)
            else nn.Identity()
        )

    def forward(self, x):
        """
        Forward pass through Conv layer.
        
        Applies convolution, batch normalization, and activation in sequence.
        
        Args:
            x (Tensor): Input tensor.
            
        Returns:
            Tensor: Output after convolution, batch normalization, and activation.
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """
        Forward pass with fused operations (without batch normalization).
        
        Used for model optimization when batch normalization can be fused with convolution.
        
        Args:
            x (Tensor): Input tensor.
            
        Returns:
            Tensor: Output after convolution and activation (without batch normalization).
        """
        return self.act(self.conv(x))


class DWConv(Conv):
    """
    Depth-wise convolution layer.
    
    A special case of grouped convolution where the number of groups equals the number of input channels,
    which significantly reduces computation compared to standard convolution.
    """

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """
        Initialize Depth-wise convolution with specified parameters.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size. Default is 1.
            s (int): Stride. Default is 1.
            d (int): Dilation factor. Default is 1.
            act (bool or nn.Layer): Activation type. Default is True.
        """
        # Use greatest common divisor of c1 and c2 as the number of groups
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class BottleNeck(nn.Layer):
    """
    Standard bottleneck module with optional shortcut connection.
    
    This module implements a bottleneck architecture that reduces channel dimensions,
    applies convolutions, and then expands back, which is efficient for deep networks.
    """

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """
        Initialize bottleneck module with configurable parameters.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            shortcut (bool): Whether to use shortcut connection. Default is True.
            g (int): Number of groups for grouped convolution. Default is 1.
            k (tuple): Kernel sizes for the two convolutions. Default is (3, 3).
            e (float): Channel expansion factor for hidden channels. Default is 0.5.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels calculated using expansion factor
        self.cv1 = Conv(c1, c_, k[0], 1)  # first convolution with kernel k[0]
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)  # second convolution with kernel k[1]
        self.add = shortcut and c1 == c2  # whether to use shortcut connection (only if input and output channels match)

    def forward(self, x):
        """
        Forward pass through the bottleneck module.
        
        Args:
            x (Tensor): Input tensor.
            
        Returns:
            Tensor: Output tensor after bottleneck operations, with optional residual connection.
        """
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Layer):
    """
    Faster implementation of CSP (Cross Stage Partial) Bottleneck with 2 convolutions.
    
    This module improves efficiency by using a more optimized structure compared to
    standard CSP bottlenecks, making it suitable for real-time applications.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """
        Initialize C2f module with specified parameters.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of bottleneck blocks. Default is 1.
            shortcut (bool): Whether to use shortcut connections in bottlenecks. Default is False.
            g (int): Number of groups for grouped convolution. Default is 1.
            e (float): Channel expansion factor for hidden channels. Default is 0.5.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # input convolution
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)  # output convolution
        self.m = nn.LayerList(
            BottleNeck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )  # bottleneck modules

    def forward(self, x):
        """
        Forward pass through C2f layer using chunk operation.
        
        Args:
            x (Tensor): Input tensor.
            
        Returns:
            Tensor: Output tensor after C2f operations.
        """
        y = paddle.chunk(self.cv1(x), chunks=2, axis=1)  # split input into two parts along channel dimension
        y = list(y)
        y.extend(m(y[-1]) for m in self.m)  # apply bottleneck modules to the second part
        return self.cv2(paddle.concat(y, axis=1))  # concatenate all parts and apply output convolution

    def forward_split(self, x):
        """
        Alternative forward pass using split operation instead of chunk.
        
        This implementation may be more efficient on some hardware.
        
        Args:
            x (Tensor): Input tensor.
            
        Returns:
            Tensor: Output tensor after C2f operations.
        """
        y = paddle.split(self.cv1(x), num_or_sections=[self.c, self.c], axis=1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(paddle.concat(y, axis=1))


class C3k2(C2f):
    """
    Enhanced implementation of CSP Bottleneck with 2 convolutions.
    
    This module extends C2f by optionally using C3k blocks instead of standard bottlenecks,
    providing more flexibility in feature extraction.
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """
        Initialize C3k2 module with specified parameters.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of bottleneck blocks. Default is 1.
            c3k (bool): Whether to use C3k blocks instead of standard bottlenecks. Default is False.
            e (float): Channel expansion factor. Default is 0.5.
            g (int): Number of groups for grouped convolution. Default is 1.
            shortcut (bool): Whether to use shortcut connections. Default is True.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        # Replace bottleneck modules with either C3k or standard BottleNeck modules
        self.m = nn.LayerList(
            C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else BottleNeck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )


class C3k(CSPLayer):
    """
    CSP bottleneck module with customizable kernel sizes.
    
    This module extends the standard CSP bottleneck by allowing custom kernel sizes,
    which can be useful for capturing different scales of features.
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """
        Initialize C3k module with specified parameters.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of bottleneck blocks. Default is 1.
            shortcut (bool): Whether to use shortcut connections. Default is True.
            g (int): Number of groups for grouped convolution. Default is 1.
            e (float): Channel expansion factor. Default is 0.5.
            k (int): Kernel size for bottleneck convolutions. Default is 3.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # Create sequential bottleneck modules with specified kernel size
        self.m = nn.Sequential(
            *(BottleNeck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n))
        )


class Attention(nn.Layer):
    """
    Multi-head self-attention module for spatial feature processing.
    
    This module implements a variant of self-attention mechanism that operates on
    spatial features, allowing the network to capture long-range dependencies.
    
    Args:
        dim (int): Input feature dimension (number of channels).
        num_heads (int): Number of attention heads. Default is 8.
        attn_ratio (float): Ratio determining the key dimension relative to head dimension. Default is 0.5.
    
    Attributes:
        num_heads (int): Number of attention heads.
        head_dim (int): Dimension of each attention head.
        key_dim (int): Dimension of the attention keys.
        scale (float): Scaling factor for attention scores to prevent gradient explosion.
        qkv (Conv): Convolutional layer for computing query, key, and value projections.
        proj (Conv): Convolutional layer for final projection of attended features.
        pe (Conv): Depthwise convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """
        Initialize multi-head attention module with specified parameters.
        
        Args:
            dim (int): Input feature dimension (number of channels).
            num_heads (int): Number of attention heads. Default is 8.
            attn_ratio (float): Ratio determining the key dimension relative to head dimension. Default is 0.5.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # dimension per head
        self.key_dim = int(self.head_dim * attn_ratio)  # key dimension (reduced by attn_ratio)
        self.scale = self.key_dim**-0.5  # scaling factor for attention scores
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2  # total dimension for query, key, and value
        self.qkv = Conv(dim, h, 1, act=False)  # projection for query, key, value
        self.proj = Conv(dim, dim, 1, act=False)  # final projection
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)  # positional encoding using depthwise convolution

    def forward(self, x):
        """
        Forward pass of the Attention module.
        
        Args:
            x (Tensor): Input tensor of shape [B, C, H, W].
            
        Returns:
            Tensor: Output tensor after self-attention of shape [B, C, H, W].
        """
        B, C, H, W = x.shape
        N = H * W  # number of spatial locations
        
        # Compute query, key, value projections
        qkv = self.qkv(x)
        qkv = paddle.reshape(
            qkv, [B, self.num_heads, self.key_dim * 2 + self.head_dim, N]
        )
        q, k, v = paddle.split(qkv, [self.key_dim, self.key_dim, self.head_dim], axis=2)

        # Compute attention scores and apply attention
        attn = paddle.matmul(q.transpose([0, 1, 3, 2]), k) * self.scale  # [B, num_heads, N, N]
        attn = nn.functional.softmax(attn, axis=-1)  # normalize attention weights
        
        # Apply attention to values and reshape
        x = paddle.matmul(v, attn.transpose([0, 1, 3, 2]))  # [B, num_heads, head_dim, N]
        x = paddle.reshape(x, [B, C, H, W]) + self.pe(paddle.reshape(v, [B, C, H, W]))  # add positional encoding
        x = self.proj(x)  # final projection
        return x


class PSABlock(nn.Layer):
    """
    Position-Sensitive Attention block combining self-attention with feed-forward networks.
    
    This block implements a transformer-like architecture with self-attention and feed-forward
    network components, adapted for convolutional neural networks. It enhances the model's
    ability to capture long-range dependencies while maintaining spatial information.
    
    Attributes:
        attn (Attention): Multi-head self-attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Whether to use residual connections.
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """
        Initialize PSABlock with specified parameters.
        
        Args:
            c (int): Number of input/output channels.
            attn_ratio (float): Ratio for attention key dimension. Default is 0.5.
            num_heads (int): Number of attention heads. Default is 4.
            shortcut (bool): Whether to use residual connections. Default is True.
        """
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)  # self-attention module
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))  # feed-forward network
        self.add = shortcut  # whether to use residual connections

    def forward(self, x):
        """
        Forward pass through PSABlock.
        
        Applies self-attention followed by feed-forward network, with optional residual connections.
        
        Args:
            x (Tensor): Input tensor.
            
        Returns:
            Tensor: Output tensor after attention and feed-forward processing.
        """
        x = x + self.attn(x) if self.add else self.attn(x)  # attention with optional residual
        x = x + self.ffn(x) if self.add else self.ffn(x)  # feed-forward with optional residual
        return x


class C2PSA(nn.Layer):
    """
    C2PSA module combining CSP (Cross Stage Partial) architecture with Position-Sensitive Attention.
    
    This module enhances feature extraction by combining the efficiency of CSP architecture
    with the long-range dependency modeling capability of position-sensitive attention.
    It splits the input into two branches, processes one branch with PSA blocks,
    and then combines them back.
    
    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): Input convolution layer.
        cv2 (Conv): Output convolution layer.
        m (nn.Sequential): Sequential container of PSABlock modules.
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """
        Initialize C2PSA module with specified parameters.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels (must equal c1).
            n (int): Number of PSABlock modules. Default is 1.
            e (float): Channel reduction factor for hidden channels. Default is 0.5.
        """
        super().__init__()
        assert c1 == c2, "Input and output channels must be equal for C2PSA"
        self.c = int(c1 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # input convolution
        self.cv2 = Conv(2 * self.c, c1, 1)  # output convolution

        # Create sequential PSABlock modules
        self.m = nn.Sequential(
            *[
                PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64)
                for _ in range(n)
            ]
        )

    def forward(self, x):
        """
        Forward pass through C2PSA module.
        
        Splits the input into two branches, processes one branch with PSA blocks,
        and then combines them back.
        
        Args:
            x (Tensor): Input tensor.
            
        Returns:
            Tensor: Output tensor after C2PSA processing.
        """
        a, b = paddle.split(self.cv1(x), num_or_sections=[self.c, self.c], axis=1)  # split along channel dimension
        b = self.m(b)  # apply PSA blocks to second branch
        return self.cv2(paddle.concat([a, b], axis=1))  # concatenate branches and apply output convolution


@register
@serializable
class YOLO11CSPDarkNet(nn.Layer):
    """
    YOLO11 CSPDarkNet backbone network architecture.
    
    This backbone is designed for YOLO11 object detection model, implementing an
    enhanced version of CSPDarkNet with various advanced modules like C3k2, SPPF, and C2PSA.
    The network follows a hierarchical structure with increasing receptive field and
    decreasing spatial resolution.
    
    Structure follows:
    [from, repeats, module, args]
    [[-1, 1, Conv, [64, 3, 2]],      # P1/2  - Initial stem convolution
     [-1, 1, Conv, [128, 3, 2]],     # P2/4  - Downsampling to 1/4 resolution
     [-1, 2, C3k2, [256, False]],    # C3k2  - Feature extraction with C3k2 blocks
     [-1, 1, Conv, [256, 3, 2]],     # P3/8  - Downsampling to 1/8 resolution
     [-1, 2, C3k2, [512, False]],    # C3k2  - Feature extraction with C3k2 blocks
     [-1, 1, Conv, [512, 3, 2]],     # P4/16 - Downsampling to 1/16 resolution
     [-1, 2, C3k2, [512, True]],     # C3k2  - Feature extraction with C3k2 blocks (with C3k)
     [-1, 1, Conv, [1024, 3, 2]],    # P5/32 - Downsampling to 1/32 resolution
     [-1, 2, C3k2, [1024, True]],    # C3k2  - Feature extraction with C3k2 blocks (with C3k)
     [-1, 1, SPPF, [1024, 5]],       # SPPF  - Spatial Pyramid Pooling - Fast
     [-1, 2, C2PSA, [1024]]]         # C2PSA - Position-Sensitive Attention blocks
    """

    __shared__ = ["depth_mult", "width_mult", "max_channels", "act", "trt"]

    # in_channels, out_channels, num_blocks, use_c3k, use_sppf, use_c2psa
    arch_settings = [
        [64, 128, 0, False, False, False],  # P2/4  - First downsampling layer
        [128, 256, 2, False, False, False],  # C3k2  - First feature extraction block
        [256, 256, 0, False, False, False],  # P3/8  - Second downsampling layer
        [256, 512, 2, False, False, False],  # C3k2  - Second feature extraction block
        [512, 512, 0, False, False, False],  # P4/16 - Third downsampling layer
        [512, 512, 2, True, False, False],   # C3k2  - Third feature extraction block (with C3k)
        [512, 1024, 0, False, False, False], # P5/32 - Fourth downsampling layer
        [1024, 1024, 2, True, False, False], # C3k2  - Fourth feature extraction block (with C3k)
        [1024, 1024, 0, False, True, False], # SPPF  - Spatial Pyramid Pooling - Fast
        [1024, 1024, 2, False, False, True], # C2PSA - Position-Sensitive Attention blocks
    ]

    def __init__(
        self,
        depth_mult=1.0,
        width_mult=1.0,
        max_channels=1024,
        depthwise=False,
        act="silu",
        trt=False,
        return_idx=[2, 3, 4],
    ):
        """
        Initialize YOLO11CSPDarkNet backbone with specified parameters.
        
        Args:
            depth_mult (float): Depth multiplier to scale number of layers. Default is 1.0.
            width_mult (float): Width multiplier to scale number of channels. Default is 1.0.
            max_channels (int): Maximum number of channels in any layer. Default is 1024.
            depthwise (bool): Whether to use depthwise separable convolutions. Default is False.
            act (str): Activation function type. Default is "silu".
            trt (bool): Whether to use TensorRT. Default is False.
            return_idx (list): Indices of stages to return for feature pyramid. Default is [2, 3, 4].
        """
        super(YOLO11CSPDarkNet, self).__init__()
        self.return_idx = return_idx  # indices of stages to return for feature pyramid
        self.Conv = DWConv if depthwise else Conv  # convolution type based on depthwise flag
        self.max_channels = max_channels  # maximum number of channels

        # Initial stem convolution
        base_channels = int(64 * width_mult)
        self.stem = self.Conv(3, base_channels, k=3, s=2, act=act)

        _out_channels = [base_channels]
        layers_num = 1
        self.csp_dark_blocks = []

        # Build network stages according to arch_settings
        for i, (
            in_channels,
            out_channels,
            num_blocks,
            use_c3k,
            use_sppf,
            use_c2psa,
        ) in enumerate(self.arch_settings):
            # Scale channels according to width_mult
            in_channels = int(in_channels * width_mult)
            out_channels = int(min(out_channels * width_mult, self.max_channels))  # Apply max_channels constraint
            _out_channels.append(out_channels)
            
            # Scale number of blocks according to depth_mult
            num_blocks = max(round(num_blocks * depth_mult), 1) if num_blocks > 0 else 0
            stage = []

            # Add Conv layer for downsampling
            conv_layer = self.add_sublayer(
                "layers{}.stage{}.conv_layer".format(layers_num, i + 1),
                self.Conv(in_channels, out_channels, 3, 2, act=act),
            )
            stage.append(conv_layer)
            layers_num += 1

            # Add C3k2 or C2PSA layer if num_blocks > 0
            if num_blocks > 0:
                if use_c2psa:
                    block = self.add_sublayer(
                        "layers{}.stage{}.c2psa".format(layers_num, i + 1),
                        C2PSA(out_channels, out_channels, n=num_blocks),
                    )
                else:
                    block = self.add_sublayer(
                        "layers{}.stage{}.c3k2".format(layers_num, i + 1),
                        C3k2(
                            out_channels,
                            out_channels,
                            n=num_blocks,
                            c3k=use_c3k,
                            e=0.25 if not use_c3k else 0.5,
                        ),
                    )
                stage.append(block)
                layers_num += 1

            # Add SPPF layer if specified
            if use_sppf:
                sppf = self.add_sublayer(
                    "layers{}.stage{}.sppf".format(layers_num, i + 1),
                    SPPFLayer(out_channels, out_channels, ksize=5, bias=False, act=act),
                )
                stage.append(sppf)
                layers_num += 1

            self.csp_dark_blocks.append(nn.Sequential(*stage))

        # Set output channels and strides for feature pyramid
        self._out_channels = [_out_channels[i] for i in self.return_idx]
        self.strides = [[2, 4, 8, 16, 32][i] for i in self.return_idx]

    def forward(self, inputs):
        """
        Forward pass through YOLO11CSPDarkNet backbone.
        
        Args:
            inputs (dict): Input dictionary containing 'image' key with input tensor.
            
        Returns:
            list: List of feature maps at different scales for feature pyramid.
        """
        x = inputs["image"]  # extract image from input dictionary
        outputs = []
        
        # Apply stem convolution
        x = self.stem(x)
        
        # Apply each stage and collect outputs at specified return_idx
        for i, layer in enumerate(self.csp_dark_blocks):
            x = layer(x)
            if i + 1 in self.return_idx:
                outputs.append(x)
                
        return outputs

    @property
    def out_shape(self):
        """
        Return output shapes for each returned feature map.
        
        Returns:
            list: List of ShapeSpec objects containing channel and stride information.
        """
        return [
            ShapeSpec(channels=c, stride=s)
            for c, s in zip(self._out_channels, self.strides)
        ]
