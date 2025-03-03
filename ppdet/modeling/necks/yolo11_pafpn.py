# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS.
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register, serializable
from ppdet.modeling.backbones.csp_darknet import get_activation
from ppdet.modeling.shape_spec import ShapeSpec
from ppdet.modeling.backbones.yolo11_csp_darknet import Conv, C3k2

__all__ = ["YOLO11CSPPAN"]


@register
@serializable
class YOLO11CSPPAN(nn.Layer):
    """
    YOLO11 Path Aggregation Feature Pyramid Network (PAFPN).

    This module implements the neck component of the YOLO11 architecture, which combines
    features from different scales using both top-down (FPN) and bottom-up (PAN) pathways.
    It enhances multi-scale feature fusion for better object detection across various scales.
    """

    __shared__ = ["depth_mult", "width_mult", "max_channels", "act", "trt"]

    def __init__(
        self,
        depth_mult=1.0,
        width_mult=1.0,
        max_channels=1024,
        in_channels=[256, 512, 1024],
        in0_channels=[128, 256, 512],
        depthwise=False,
        act="silu",
        trt=False,
    ):
        """
        Initialize the YOLO11CSPPAN module.

        Args:
            depth_mult (float): Depth multiplier for controlling the model's capacity.
            width_mult (float): Width multiplier for controlling the model's capacity.
            max_channels (int): Maximum number of channels in any layer.
            in_channels (list): List of input channel dimensions from the backbone
                                (typically [256, 512, 1024] for small, medium, and large feature maps).
            depthwise (bool): Whether to use depthwise separable convolutions for reduced computation.
            act (str): Activation function type, default is 'silu' (Sigmoid Linear Unit).
            trt (bool): Whether to use TensorRT compatibility mode.
        """
        super(YOLO11CSPPAN, self).__init__()

        in_channels = [int(min(c * width_mult, max_channels)) for c in in_channels]
        self._out_channels = in_channels

        self.fpn_p4 = C3k2(
            int(in0_channels[1] + in_channels[2]),
            int(in_channels[1]),
            round(2 * depth_mult),
            c3k=False,
        )

        self.fpn_p3 = C3k2(
            int(in0_channels[0] + in_channels[1]),
            int(in_channels[0]),
            round(2 * depth_mult),
            c3k=False,
        )

        self.down_conv2 = Conv(
            int(in_channels[0]), int(in_channels[0]), 3, s=2, act=get_activation(act)
        )

        self.pan_n3 = C3k2(
            int(in_channels[0] + in_channels[1]),
            int(in_channels[1]),
            round(2 * depth_mult),
            c3k=False,
        )

        self.down_conv1 = Conv(
            int(in_channels[1]), int(in_channels[1]), 3, s=2, act=get_activation(act)
        )

        self.pan_n4 = C3k2(
            int(in_channels[1] + in_channels[2]),
            int(in_channels[2]),
            round(2 * depth_mult),
            c3k=True,
        )

    def forward(self, feats, for_mot=False):
        """
        Forward pass of the YOLO11CSPPAN.

        Args:
            feats (list): List of feature maps [C3, C4, C5] from the backbone network,
                          where C3 is the smallest scale (highest resolution),
                          and C5 is the largest scale (lowest resolution).
            for_mot (bool): Flag for Multiple Object Tracking mode, not used in this implementation.

        Returns:
            list: List of processed feature maps [P3, P4, P5] for detection heads,
                  where P3 is the smallest scale (highest resolution) feature map,
                  and P5 is the largest scale (lowest resolution) feature map.
        """
        [c3, c4, c5] = feats

        up_feat1 = F.interpolate(c5, scale_factor=2.0, mode="nearest")
        f_concat1 = paddle.concat([up_feat1, c4], 1)
        f_out1 = self.fpn_p4(f_concat1)

        up_feat2 = F.interpolate(f_out1, scale_factor=2.0, mode="nearest")
        f_concat2 = paddle.concat([up_feat2, c3], 1)
        f_out0 = self.fpn_p3(f_concat2)

        down_feat1 = self.down_conv2(f_out0)
        p_concat1 = paddle.concat([down_feat1, f_out1], 1)
        pan_out1 = self.pan_n3(p_concat1)

        down_feat2 = self.down_conv1(pan_out1)
        p_concat2 = paddle.concat([down_feat2, c5], 1)
        pan_out0 = self.pan_n4(p_concat2)

        return [f_out0, pan_out1, pan_out0]

    @classmethod
    def from_config(cls, cfg, input_shape):
        """
        Create configuration for YOLO11CSPPAN from model config and input shapes.

        Args:
            cfg: Model configuration parameters.
            input_shape: Shapes of the input feature maps from the backbone.

        Returns:
            dict: Configuration dictionary with input channel dimensions.
        """
        return {
            "in0_channels": [i.channels for i in input_shape],
        }

    @property
    def out_shape(self):
        """
        Get the output shapes of the YOLO11CSPPAN.

        Returns:
            list: List of ShapeSpec objects describing the channel dimensions of output feature maps.
        """
        return [ShapeSpec(channels=c) for c in self._out_channels]
