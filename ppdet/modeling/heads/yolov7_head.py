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

import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from ppdet.core.workspace import register
from ..backbones.yolov7_elannet import ImplicitA, ImplicitM
from ppdet.modeling.layers import MultiClassNMS

__all__ = ['YOLOv7Head']


@register
class YOLOv7Head(nn.Layer):
    __shared__ = [
        'num_classes', 'data_format', 'use_aux', 'use_implicit', 'trt',
        'exclude_nms', 'exclude_post_process'
    ]
    __inject__ = ['loss', 'nms']

    def __init__(self,
                 num_classes=80,
                 in_channels=[256, 512, 1024],
                 anchors=[[12, 16], [19, 36], [40, 28], [36, 75], [76, 55],
                          [72, 146], [142, 110], [192, 243], [459, 401]],
                 anchor_masks=[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                 stride=[8, 16, 32],
                 use_aux=False,
                 use_implicit=False,
                 loss='YOLOv7Loss',
                 data_format='NCHW',
                 nms='MultiClassNMS',
                 trt=False,
                 exclude_post_process=False,
                 exclude_nms=False):
        """
        Head for YOLOv7

        Args:
            num_classes (int): number of foreground classes
            in_channels (int): channels of input features
            anchors (list): anchors
            anchor_masks (list): anchor masks
            stride (list): strides
            use_aux (bool): whether to use Aux Head, only in P6 models
            use_implicit (bool): whether to use ImplicitA and ImplicitM
            loss (object): YOLOv7Loss instance
            data_format (str): nms format, NCHW or NHWC
            nms (object): MultiClassNMS instance
            trt (bool): whether to use trt infer
            exclude_nms (bool): whether to use exclude_nms for speed test
        """
        super(YOLOv7Head, self).__init__()
        assert len(in_channels) > 0, "in_channels length should > 0"
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.parse_anchor(anchors, anchor_masks)
        self.anchors = paddle.to_tensor(self.anchors, dtype='int32')
        self.anchor_levels = len(self.anchors)

        self.stride = stride
        self.use_aux = use_aux
        self.use_implicit = use_implicit
        self.loss = loss
        self.data_format = data_format
        self.nms = nms
        if isinstance(self.nms, MultiClassNMS) and trt:
            self.nms.trt = trt
        self.exclude_nms = exclude_nms
        self.exclude_post_process = exclude_post_process

        self.num_anchor = len(self.anchors[0])  # self.na
        self.num_out_ch = self.num_classes + 5  # self.no

        self.yolo_outputs = []
        if self.use_aux:
            self.yolo_outputs_aux = []
        if self.use_implicit:
            self.ia, self.im = [], []
        self.num_levels = len(self.anchors)
        for i in range(self.num_levels):
            num_filters = self.num_anchor * self.num_out_ch
            name = 'yolo_output.{}'.format(i)
            conv = nn.Conv2D(
                in_channels=self.in_channels[i],
                out_channels=num_filters,
                kernel_size=1,
                stride=1,
                padding=0,
                data_format=data_format,
                bias_attr=ParamAttr(regularizer=L2Decay(0.)))
            conv.skip_quant = True
            yolo_output = self.add_sublayer(name, conv)
            self.yolo_outputs.append(yolo_output)

            if self.use_aux:
                name_aux = 'yolo_output_aux.{}'.format(i)
                conv_aux = nn.Conv2D(
                    in_channels=self.in_channels[i + self.num_levels],
                    out_channels=num_filters,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    data_format=data_format,
                    bias_attr=ParamAttr(regularizer=L2Decay(0.)))
                conv_aux.skip_quant = True
                yolo_output_aux = self.add_sublayer(name_aux, conv_aux)
                self.yolo_outputs_aux.append(yolo_output_aux)

            if self.use_implicit:
                ia = ImplicitA(self.in_channels[i])
                yolo_output_ia = self.add_sublayer(
                    'yolo_output_ia.{}'.format(i), ia)
                self.ia.append(yolo_output_ia)

                im = ImplicitM(num_filters)
                yolo_output_im = self.add_sublayer(
                    'yolo_output_im.{}'.format(i), im)
                self.im.append(yolo_output_im)

        self._initialize_biases(self.yolo_outputs)
        if self.use_aux:
            self._initialize_biases(self.yolo_outputs_aux)

    def fuse(self):
        if self.use_implicit:
            # fuse ImplicitA and Convolution
            for i in range(len(self.yolo_outputs)):
                c1, c2, _, _ = self.yolo_outputs[
                    i].weight.shape  # [255, 256, 1, 1]
                c1_, c2_, _, _ = self.ia[i].ia.shape  # [1, 256, 1, 1]
                cc = paddle.matmul(self.yolo_outputs[i].weight.reshape(
                    [c1, c2]), self.ia[i].ia.reshape([c2_, c1_])).squeeze(1)
                self.yolo_outputs[i].bias.set_value(self.yolo_outputs[i].bias +
                                                    cc)

            # fuse ImplicitM and Convolution
            for i in range(len(self.yolo_outputs)):
                c1, c2, _, _ = self.im[i].im.shape  # [1, 255, 1, 1]
                self.yolo_outputs[i].bias.set_value(self.yolo_outputs[i].bias *
                                                    self.im[i].im.reshape([c2]))
                self.yolo_outputs[i].weight.set_value(
                    self.yolo_outputs[i].weight * paddle.transpose(
                        self.im[i].im, [1, 0, 2, 3]))

    def _initialize_biases(self, convs):
        # initialize biases, see https://arxiv.org/abs/1708.02002 section 3.3
        for i, conv in enumerate(convs):
            b = conv.bias.numpy().reshape([3, -1])  # [255] to [3,85]
            b[:, 4] += math.log(8 / (640 / self.stride[i])**2)
            b[:, 5:] += math.log(0.6 / (self.num_classes - 0.99))
            conv.bias.set_value(b.reshape([-1]))

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def parse_anchor(self, anchors, anchor_masks):
        self.anchors = [[anchors[i] for i in mask] for mask in anchor_masks]
        self.mask_anchors = []
        anchor_num = len(anchors)
        for masks in anchor_masks:
            self.mask_anchors.append([])
            for mask in masks:
                assert mask < anchor_num, "anchor mask index overflow"
                self.mask_anchors[-1].extend(anchors[mask])

    def forward(self, feats, targets=None):
        yolo_outputs = []
        if self.training and self.use_aux:
            yolo_outputs_aux = []
        for i in range(self.num_levels):
            if self.training and self.use_implicit:
                yolo_output = self.im[i](self.yolo_outputs[i](self.ia[i](feats[
                    i])))
            else:
                yolo_output = self.yolo_outputs[i](feats[i])
            if self.data_format == 'NHWC':
                yolo_output = paddle.transpose(yolo_output, [0, 3, 1, 2])
            yolo_outputs.append(yolo_output)

            if self.training and self.use_aux:
                yolo_output_aux = self.yolo_outputs_aux[i](feats[
                    i + self.num_levels])
                yolo_outputs_aux.append(yolo_output_aux)

        if self.training:
            if self.use_aux:
                return self.loss(yolo_outputs + yolo_outputs_aux, targets,
                                 self.anchors)
            else:
                return self.loss(yolo_outputs, targets, self.anchors)
        else:
            return yolo_outputs

    def make_grid(self, nx, ny, anchor):
        yv, xv = paddle.meshgrid([
            paddle.arange(
                ny, dtype='int32'), paddle.arange(
                    nx, dtype='int32')
        ])
        grid = paddle.stack((xv, yv), axis=2).reshape([1, 1, ny, nx, 2])
        anchor_grid = anchor.reshape([1, self.num_anchor, 1, 1, 2])
        return grid, anchor_grid

    def postprocessing_by_level(self, head_out, stride, anchor, ny, nx):
        grid, anchor_grid = self.make_grid(nx, ny, anchor)
        out = F.sigmoid(head_out)
        xy = (out[..., 0:2] * 2. - 0.5 + grid) * stride
        wh = (out[..., 2:4] * 2)**2 * anchor_grid
        lt_xy = (xy - wh / 2.)
        rb_xy = (xy + wh / 2.)
        bboxes = paddle.concat((lt_xy, rb_xy), axis=-1)
        scores = out[..., 5:] * out[..., 4].unsqueeze(-1)
        return bboxes, scores

    def post_process(self, head_outs, img_shape, scale_factor):
        bbox_list, score_list = [], []

        for i, head_out in enumerate(head_outs):
            _, _, ny, nx = head_out.shape
            head_out = head_out.reshape(
                [-1, self.num_anchor, self.num_out_ch, ny, nx]).transpose(
                    [0, 1, 3, 4, 2])
            # head_out.shape [bs, self.num_anchor, ny, nx, self.num_out_ch]

            bbox, score = self.postprocessing_by_level(head_out, self.stride[i],
                                                       self.anchors[i], ny, nx)
            bbox = bbox.reshape([-1, self.num_anchor * ny * nx, 4])
            score = score.reshape(
                [-1, self.num_anchor * ny * nx, self.num_classes]).transpose(
                    [0, 2, 1])
            bbox_list.append(bbox)
            score_list.append(score)
        pred_bboxes = paddle.concat(bbox_list, axis=1)
        pred_scores = paddle.concat(score_list, axis=-1)

        if self.exclude_post_process:
            return paddle.concat(
                [pred_bboxes, pred_scores.transpose([0, 2, 1])], axis=-1)
        else:
            # scale bbox to origin
            scale_factor = scale_factor.flip(-1).tile([1, 2]).unsqueeze(1)
            pred_bboxes /= scale_factor
            if self.exclude_nms:
                # `exclude_nms=True` just use in benchmark
                return pred_bboxes, pred_scores
            else:
                bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
                return bbox_pred, bbox_num
