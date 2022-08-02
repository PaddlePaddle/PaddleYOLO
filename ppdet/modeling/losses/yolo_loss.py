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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register

from ..bbox_utils import decode_yolo, xywh2xyxy, batch_iou_similarity, bbox_iou

__all__ = ['YOLOv3Loss', 'YOLOv5Loss', 'YOLOv7Loss']


def bbox_transform(pbox, anchor, downsample):
    pbox = decode_yolo(pbox, anchor, downsample)
    pbox = xywh2xyxy(pbox)
    return pbox


@register
class YOLOv3Loss(nn.Layer):

    __inject__ = ['iou_loss', 'iou_aware_loss']
    __shared__ = ['num_classes']

    def __init__(self,
                 num_classes=80,
                 ignore_thresh=0.7,
                 label_smooth=False,
                 downsample=[32, 16, 8],
                 scale_x_y=1.,
                 iou_loss=None,
                 iou_aware_loss=None):
        """
        YOLOv3Loss layer

        Args:
            num_calsses (int): number of foreground classes
            ignore_thresh (float): threshold to ignore confidence loss
            label_smooth (bool): whether to use label smoothing
            downsample (list): downsample ratio for each detection block
            scale_x_y (float): scale_x_y factor
            iou_loss (object): IoULoss instance
            iou_aware_loss (object): IouAwareLoss instance  
        """
        super(YOLOv3Loss, self).__init__()
        self.num_classes = num_classes
        self.ignore_thresh = ignore_thresh
        self.label_smooth = label_smooth
        self.downsample = downsample
        self.scale_x_y = scale_x_y
        self.iou_loss = iou_loss
        self.iou_aware_loss = iou_aware_loss
        self.distill_pairs = []

    def obj_loss(self, pbox, gbox, pobj, tobj, anchor, downsample):
        # pbox
        pbox = decode_yolo(pbox, anchor, downsample)
        pbox = xywh2xyxy(pbox)
        pbox = paddle.concat(pbox, axis=-1)
        b = pbox.shape[0]
        pbox = pbox.reshape((b, -1, 4))
        # gbox
        gxy = gbox[:, :, 0:2] - gbox[:, :, 2:4] * 0.5
        gwh = gbox[:, :, 0:2] + gbox[:, :, 2:4] * 0.5
        gbox = paddle.concat([gxy, gwh], axis=-1)

        iou = batch_iou_similarity(pbox, gbox)
        iou.stop_gradient = True
        iou_max = iou.max(2)  # [N, M1]
        iou_mask = paddle.cast(iou_max <= self.ignore_thresh, dtype=pbox.dtype)
        iou_mask.stop_gradient = True

        pobj = pobj.reshape((b, -1))
        tobj = tobj.reshape((b, -1))
        obj_mask = paddle.cast(tobj > 0, dtype=pbox.dtype)
        obj_mask.stop_gradient = True

        loss_obj = F.binary_cross_entropy_with_logits(
            pobj, obj_mask, reduction='none')
        loss_obj_pos = (loss_obj * tobj)
        loss_obj_neg = (loss_obj * (1 - obj_mask) * iou_mask)
        return loss_obj_pos + loss_obj_neg

    def cls_loss(self, pcls, tcls):
        if self.label_smooth:
            delta = min(1. / self.num_classes, 1. / 40)
            pos, neg = 1 - delta, delta
            # 1 for positive, 0 for negative
            tcls = pos * paddle.cast(
                tcls > 0., dtype=tcls.dtype) + neg * paddle.cast(
                    tcls <= 0., dtype=tcls.dtype)

        loss_cls = F.binary_cross_entropy_with_logits(
            pcls, tcls, reduction='none')
        return loss_cls

    def yolov3_loss(self, p, t, gt_box, anchor, downsample, scale=1.,
                    eps=1e-10):
        na = len(anchor)
        b, c, h, w = p.shape
        if self.iou_aware_loss:
            ioup, p = p[:, 0:na, :, :], p[:, na:, :, :]
            ioup = ioup.unsqueeze(-1)
        p = p.reshape((b, na, -1, h, w)).transpose((0, 1, 3, 4, 2))
        x, y = p[:, :, :, :, 0:1], p[:, :, :, :, 1:2]
        w, h = p[:, :, :, :, 2:3], p[:, :, :, :, 3:4]
        obj, pcls = p[:, :, :, :, 4:5], p[:, :, :, :, 5:]
        self.distill_pairs.append([x, y, w, h, obj, pcls])

        t = t.transpose((0, 1, 3, 4, 2))
        tx, ty = t[:, :, :, :, 0:1], t[:, :, :, :, 1:2]
        tw, th = t[:, :, :, :, 2:3], t[:, :, :, :, 3:4]
        tscale = t[:, :, :, :, 4:5]
        tobj, tcls = t[:, :, :, :, 5:6], t[:, :, :, :, 6:]

        tscale_obj = tscale * tobj
        loss = dict()

        x = scale * F.sigmoid(x) - 0.5 * (scale - 1.)
        y = scale * F.sigmoid(y) - 0.5 * (scale - 1.)

        if abs(scale - 1.) < eps:
            loss_x = F.binary_cross_entropy(x, tx, reduction='none')
            loss_y = F.binary_cross_entropy(y, ty, reduction='none')
            loss_xy = tscale_obj * (loss_x + loss_y)
        else:
            loss_x = paddle.abs(x - tx)
            loss_y = paddle.abs(y - ty)
            loss_xy = tscale_obj * (loss_x + loss_y)

        loss_xy = loss_xy.sum([1, 2, 3, 4]).mean()

        loss_w = paddle.abs(w - tw)
        loss_h = paddle.abs(h - th)
        loss_wh = tscale_obj * (loss_w + loss_h)
        loss_wh = loss_wh.sum([1, 2, 3, 4]).mean()

        loss['loss_xy'] = loss_xy
        loss['loss_wh'] = loss_wh

        if self.iou_loss is not None:
            # warn: do not modify x, y, w, h in place
            box, tbox = [x, y, w, h], [tx, ty, tw, th]
            pbox = bbox_transform(box, anchor, downsample)
            gbox = bbox_transform(tbox, anchor, downsample)
            loss_iou = self.iou_loss(pbox, gbox)
            loss_iou = loss_iou * tscale_obj
            loss_iou = loss_iou.sum([1, 2, 3, 4]).mean()
            loss['loss_iou'] = loss_iou

        if self.iou_aware_loss is not None:
            box, tbox = [x, y, w, h], [tx, ty, tw, th]
            pbox = bbox_transform(box, anchor, downsample)
            gbox = bbox_transform(tbox, anchor, downsample)
            loss_iou_aware = self.iou_aware_loss(ioup, pbox, gbox)
            loss_iou_aware = loss_iou_aware * tobj
            loss_iou_aware = loss_iou_aware.sum([1, 2, 3, 4]).mean()
            loss['loss_iou_aware'] = loss_iou_aware

        box = [x, y, w, h]
        loss_obj = self.obj_loss(box, gt_box, obj, tobj, anchor, downsample)
        loss_obj = loss_obj.sum(-1).mean()
        loss['loss_obj'] = loss_obj
        loss_cls = self.cls_loss(pcls, tcls) * tobj
        loss_cls = loss_cls.sum([1, 2, 3, 4]).mean()
        loss['loss_cls'] = loss_cls
        return loss

    def forward(self, inputs, targets, anchors):
        np = len(inputs)
        gt_targets = [targets['target{}'.format(i)] for i in range(np)]
        gt_box = targets['gt_bbox']
        yolo_losses = dict()
        self.distill_pairs.clear()
        for x, t, anchor, downsample in zip(inputs, gt_targets, anchors,
                                            self.downsample):
            yolo_loss = self.yolov3_loss(x, t, gt_box, anchor, downsample,
                                         self.scale_x_y)
            for k, v in yolo_loss.items():
                if k in yolo_losses:
                    yolo_losses[k] += v
                else:
                    yolo_losses[k] = v

        loss = 0
        for k, v in yolo_losses.items():
            loss += v

        yolo_losses['loss'] = loss
        return yolo_losses


@register
class YOLOv5Loss(nn.Layer):
    __shared__ = ['num_classes']

    def __init__(self,
                 num_classes=80,
                 downsample_ratios=[8, 16, 32],
                 balance=[4.0, 1.0, 0.4],
                 box_weight=0.05,
                 obj_weight=1.0,
                 cls_weght=0.5,
                 bias=0.5,
                 anchor_t=4.0,
                 label_smooth_eps=0.):
        super(YOLOv5Loss, self).__init__()
        self.num_classes = num_classes
        self.balance = balance
        self.na = 3  # not len(anchors)
        self.gr = 1.0

        self.BCEcls = nn.BCEWithLogitsLoss(
            pos_weight=paddle.to_tensor([1.0]), reduction="mean")
        self.BCEobj = nn.BCEWithLogitsLoss(
            pos_weight=paddle.to_tensor([1.0]), reduction="mean")

        self.loss_weights = {
            'box': box_weight,
            'obj': obj_weight,
            'cls': cls_weght,
        }

        eps = label_smooth_eps if label_smooth_eps > 0 else 0.
        self.cls_pos_label = 1.0 - 0.5 * eps
        self.cls_neg_label = 0.5 * eps

        self.downsample_ratios = downsample_ratios
        self.bias = bias
        self.off = paddle.to_tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
            ],
            dtype='float32') * self.bias
        self.anchor_t = anchor_t

    def build_targets(self, outputs, targets, anchors):
        anchors = anchors.numpy()
        gt_nums = [len(bbox) for bbox in targets['gt_bbox']]
        nt = int(sum(gt_nums))
        na = anchors.shape[1]  # not len(anchors)
        tcls, tbox, indices, anch = [], [], [], []

        gain = np.ones(7, dtype=np.float32)  # normalized to gridspace gain
        ai = np.tile(
            np.arange(
                na, dtype=np.float32).reshape(na, 1),
            [1, nt])  # same as .repeat_interleave(nt)

        batch_size = outputs[0].shape[0]
        gt_labels = []
        for idx in range(batch_size):
            gt_num = gt_nums[idx]
            if gt_num == 0:
                continue
            gt_bbox = targets['gt_bbox'][idx][:gt_num]
            gt_class = targets['gt_class'][idx][:gt_num] * 1.0
            img_idx = np.repeat(np.array([[idx]]), gt_num, axis=0)
            gt_labels.append(
                np.concatenate(
                    (img_idx, gt_class, gt_bbox), axis=-1))
        if (len(gt_labels)):
            gt_labels = np.concatenate(gt_labels)
        else:
            gt_labels = np.zeros([0, 6])

        targets_labels = np.concatenate((np.tile(
            np.expand_dims(gt_labels, 0), [na, 1, 1]), ai[:, :, None]), 2)
        g = 0.5  # bias
        off = np.array(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
            ],
            dtype=np.float32) * g  # offsets

        for i in range(len(anchors)):
            anchor = np.array(anchors[i]) / self.downsample_ratios[i]
            gain[2:6] = np.array(
                outputs[i].shape, dtype=np.float32)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets_labels to
            t = targets_labels * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchor[:, None]
                j = np.maximum(r, 1 / r).max(2) < self.anchor_t
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = np.stack((np.ones_like(j), j, k, l, m))
                t = np.tile(t, [5, 1, 1])[j]
                offsets = (np.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets_labels[0]
                offsets = 0

            # Define
            b, c = t[:, :2].astype(np.int64).T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).astype(np.int64)
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].astype(np.int64)  # anchor indices
            gj, gi = gj.clip(0, gain[3] - 1), gi.clip(0, gain[2] - 1)
            indices.append((paddle.to_tensor(b), paddle.to_tensor(a),
                            paddle.to_tensor(
                                gj, dtype=paddle.int64), paddle.to_tensor(
                                    gi, dtype=paddle.int64)))
            tbox.append(
                paddle.to_tensor(
                    np.concatenate((gxy - gij, gwh), 1), dtype=paddle.float32))
            anch.append(paddle.to_tensor(anchor[a]))
            tcls.append(paddle.to_tensor(c))
        return tcls, tbox, indices, anch

    def yolov5_loss(self, pi, t_cls, t_box, t_indices, t_anchor, balance):
        loss = dict()
        b, a, gj, gi = t_indices  # image, anchor, gridy, gridx
        n = b.shape[0]  # number of targets
        tobj = paddle.zeros_like(pi[:, :, :, :, 4])
        loss_box = paddle.to_tensor([0.])
        loss_cls = paddle.to_tensor([0.])
        if n:
            ps = pi.gather_nd(
                paddle.concat([
                    b.reshape([-1, 1]), a.reshape([-1, 1]), gj.reshape([-1, 1]),
                    gi.reshape([-1, 1])
                ], 1))
            # Regression
            pxy = F.sigmoid(ps[:, :2]) * 2 - 0.5
            pwh = (F.sigmoid(ps[:, 2:4]) * 2)**2 * t_anchor
            pbox = paddle.concat((pxy, pwh), 1)  # predicted box # [21, 4]
            iou = bbox_iou(pbox.T, t_box.T, x1y1x2y2=False, ciou=True)
            # iou.stop_gradient = True
            loss_box = (1.0 - iou).mean()

            # Objectness
            score_iou = paddle.cast(iou.detach().clip(0), tobj.dtype)
            with paddle.no_grad():
                tobj[b, a, gj, gi] = (1.0 - self.gr
                                      ) + self.gr * score_iou  # iou ratio

            # Classification
            t = paddle.full_like(ps[:, 5:], self.cls_neg_label)
            t[range(n), t_cls] = self.cls_pos_label
            loss_cls = self.BCEcls(ps[:, 5:], t)

        obji = self.BCEobj(pi[:, :, :, :, 4], tobj)  # [4, 3, 80, 80]

        loss_obj = obji * balance

        loss['loss_box'] = loss_box * self.loss_weights['box']
        loss['loss_obj'] = loss_obj * self.loss_weights['obj']
        loss['loss_cls'] = loss_cls * self.loss_weights['cls']
        return loss

    def forward(self, inputs, targets, anchors):
        assert len(inputs) == len(anchors)
        assert len(inputs) == len(self.downsample_ratios)
        yolo_losses = dict()
        tcls, tbox, indices, anch = self.build_targets(inputs, targets, anchors)

        for i, (p_det, balance) in enumerate(zip(inputs, self.balance)):
            t_cls = tcls[i]
            t_box = tbox[i]
            t_anchor = anch[i]
            t_indices = indices[i]

            bs, ch, h, w = p_det.shape
            pi = p_det.reshape((bs, self.na, -1, h, w)).transpose(
                (0, 1, 3, 4, 2))

            yolo_loss = self.yolov5_loss(pi, t_cls, t_box, t_indices, t_anchor,
                                         balance)

            for k, v in yolo_loss.items():
                if k in yolo_losses:
                    yolo_losses[k] += v
                else:
                    yolo_losses[k] = v

        loss = 0
        for k, v in yolo_losses.items():
            loss += v

        batch_size = inputs[0].shape[0]
        num_gpus = targets.get('num_gpus', 8)
        yolo_losses['loss'] = loss * batch_size * num_gpus
        return yolo_losses


from IPython import embed


@register
class YOLOv7Loss(nn.Layer):
    __shared__ = ['num_classes']

    def __init__(self,
                 num_classes=80,
                 downsample_ratios=[8, 16, 32],
                 balance=[4.0, 1.0, 0.4],
                 box_weight=0.05,
                 cls_weght=0.3,
                 obj_weight=0.7,
                 bias=0.5,
                 anchor_t=4.0,
                 label_smooth_eps=0.):
        super(YOLOv7Loss, self).__init__()
        self.num_classes = num_classes
        self.balance = balance
        self.na = 3  # not len(anchors)
        self.gr = 1.0

        self.BCEcls = nn.BCEWithLogitsLoss(
            pos_weight=paddle.to_tensor([1.0]), reduction="mean")
        self.BCEobj = nn.BCEWithLogitsLoss(
            pos_weight=paddle.to_tensor([1.0]), reduction="mean")

        self.loss_weights = {
            'box': box_weight,
            'obj': obj_weight,
            'cls': cls_weght,
        }

        eps = label_smooth_eps if label_smooth_eps > 0 else 0.
        self.cls_pos_label = 1.0 - 0.5 * eps
        self.cls_neg_label = 0.5 * eps

        self.downsample_ratios = downsample_ratios
        self.bias = bias
        self.off = paddle.to_tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
            ],
            dtype='float32') * self.bias
        self.anchor_t = anchor_t

    def forward(self, head_outs, gt_targets, anchors):
        assert len(head_outs) == len(anchors)
        assert len(head_outs) == len(self.downsample_ratios)
        yolo_losses = dict()
        inputs = []
        for i, pi in enumerate(head_outs):
            bs, ch, h, w = pi.shape
            pi = pi.reshape((bs, self.na, ch // self.na, h, w)).transpose(
                (0, 1, 3, 4, 2))
            inputs.append(pi)
            #print(i, pi.shape, pi.sum())
            #np.save('p{}.npy'.format(i), pi)
        # 0 [8, 3, 80, 80, 85] [-91915176.]
        # 1 [8, 3, 40, 40, 85] [-23893948.]
        # 2 [8, 3, 20, 20, 85] [-6398980.]

        batch_size = head_outs[0].shape[0]
        img_idx = []
        gt_nums = [len(bbox) for bbox in gt_targets['gt_bbox']]
        for idx in range(batch_size):
            gt_num = gt_nums[idx]
            if gt_num == 0: continue
            img_idx.append(np.repeat(np.array([[idx]]), gt_num, axis=0))
        yolov7_gt_index = paddle.to_tensor(np.concatenate(img_idx), 'float32')
        yolov7_gt_class = paddle.cast(
            paddle.concat(
                gt_targets['gt_class'], axis=0), 'float32')
        yolov7_gt_bbox = paddle.cast(
            paddle.concat(
                gt_targets['gt_bbox'], axis=0), 'float32')
        try:
            targets = paddle.concat(
                [yolov7_gt_index, yolov7_gt_class, yolov7_gt_bbox], 1)
        except:
            print('concat([yolov7_gt_index, yolov7_gt_class, yolov7_gt_bbox]')
            embed()
        #print('targets: ', targets.shape, targets.sum()) # [22, 6] [-6398980.]
        #np.save('targets.npy', targets)

        ### copy
        lcls, lbox, lobj = paddle.zeros([1]), paddle.zeros([1]), paddle.zeros(
            [1])
        bs, as_, gjs, gis, targets, anchors = self.build_targets(
            inputs, targets, gt_targets['image'], anchors)
        pre_gen_gains = [
            paddle.to_tensor(pp.shape)[[3, 2, 3, 2]] for pp in inputs
        ]

        # Losses
        for i, pi in enumerate(inputs):  # layer index, layer predictions
            b, a, gj, gi = bs[i], as_[i], gjs[i], gis[
                i]  # image, anchor, gridy, gridx
            tobj = paddle.zeros_like(pi[..., 0])  # target obj #######
            n = b.shape[0]  # number of targets
            if n:
                #print('for i, pi in enumerate(inputs): p{}'.format(i), pi.sum(), pi.shape)
                ps = pi[b, a, gj,
                        gi]  # prediction subset corresponding to targets
                # Regression
                grid = paddle.stack([gi, gj], 1)
                pxy = F.sigmoid(ps[:, :2]) * 2. - 0.5
                pwh = (F.sigmoid(ps[:, 2:4]) * 2)**2 * anchors[i]
                pbox = paddle.concat([pxy, pwh], 1)  # predicted box
                selected_tbox = targets[i][:, 2:6] * pre_gen_gains[i]
                selected_tbox[:, :2] -= grid
                try:
                    iou = bbox_iou(
                        pbox.T,
                        selected_tbox.T,
                        x1y1x2y2=False,
                        ciou=True,
                        eps=1e-7)  # iou(prediction, target)
                except:
                    print(
                        'iou = bbox_iou(pbox.T, selected_tbox, x1y1x2y2=False, ciou=True)'
                    )
                    embed()
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                #tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio
                score_iou = paddle.cast(iou.detach().clip(0), tobj.dtype)
                with paddle.no_grad():
                    tobj[b, a, gj, gi] = (1.0 - self.gr
                                          ) + self.gr * score_iou  # iou ratio

                # Classification
                selected_tcls = paddle.cast(targets[i][:, 1], 'int64')  #
                if self.num_classes > 1:  # cls loss (only if multiple classes)
                    t = paddle.full_like(ps[:, 5:],
                                         self.cls_neg_label)  # targets
                    t[range(n), selected_tcls] = self.cls_pos_label
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

            obji = self.BCEobj(pi[:, :, :, :, 4], tobj)
            lobj += obji * self.balance[i]  # obj loss

        yolo_losses = dict()
        yolo_losses['loss_box'] = lbox * self.loss_weights['box']
        yolo_losses['loss_obj'] = lobj * self.loss_weights['obj']
        yolo_losses['loss_cls'] = lcls * self.loss_weights['cls']
        loss_all = yolo_losses['loss_box'] + yolo_losses[
            'loss_obj'] + yolo_losses['loss_cls']
        batch_size = head_outs[0].shape[0]
        num_gpus = gt_targets.get('num_gpus', 8)
        yolo_losses['loss'] = loss_all * batch_size * num_gpus
        return yolo_losses

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone()  #if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def box_iou(self, box1, box2):
        # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            box1 (Tensor[N, 4])
            box2 (Tensor[M, 4])
        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2
        """

        def box_area(box):
            # box = 4xn
            return (box[2] - box[0]) * (box[3] - box[1])

        area1 = box_area(box1.T)
        area2 = box_area(box2.T)

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        inter = (paddle.minimum(box1[:, None, 2:], box2[:, 2:]) -
                 paddle.maximum(box1[:, None, :2], box2[:, :2])).clip(0).prod(2)
        return inter / (area1[:, None] + area2 - inter
                        )  # iou = inter / (area1 + area2 - inter)

    def build_targets(self, p, targets, imgs, anchors):
        # [8, 3, 80, 80, 85] [8, 3, 40, 40, 85] [8, 3, 20, 20, 85], targets [22, 6], imgs [8, 3, 640, 640]
        # [-91915176.] [-23893948.] [-6398980.]
        indices, anch = self.find_3_positive(p, targets, anchors)

        matching_bs = [[] for pp in p]
        matching_as = [[] for pp in p]
        matching_gjs = [[] for pp in p]
        matching_gis = [[] for pp in p]
        matching_targets = [[] for pp in p]
        matching_anchs = [[] for pp in p]

        nl = len(p)
        for batch_idx in range(p[0].shape[0]):
            b_idx = targets[:, 0] == batch_idx
            this_target = targets[b_idx]
            if this_target.shape[0] == 0:
                continue
            txywh = this_target[:, 2:6] * imgs[batch_idx].shape[1]  # 640
            txyxy = self.xywh2xyxy(txywh)

            pxyxys, p_cls, p_obj = [], [], []
            from_which_layer = []
            all_b, all_a, all_gj, all_gi = [], [], [], []
            all_anch = []

            empty_feats_num = 0
            for i, pi in enumerate(p):
                #b, a, gj, gi = indices[i]
                idx = (indices[i][0] == batch_idx)
                if idx.sum() == 0:
                    empty_feats_num += 1
                    # print('batch_idx {} level {} empty: '.format(batch_idx, i))
                    continue
                #b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]
                b, a, gj, gi = indices[i][0][idx], indices[i][1][idx], indices[
                    i][2][idx], indices[i][3][idx]
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append(paddle.ones([len(b)]) * i)

                try:
                    fg_pred = pi[b, a, gj, gi]  #
                    if len(fg_pred.shape) == 1:
                        fg_pred = fg_pred.unsqueeze(0)
                except:
                    print('batch_idx: {}, fg_pred = pi[b, a, gj, gi]'.format(
                        batch_idx), b)
                    embed()
                p_obj.append(fg_pred[:, 4:5])
                p_cls.append(fg_pred[:, 5:])

                grid = paddle.stack([gi, gj], 1)
                pxy = (F.sigmoid(fg_pred[:, :2]) * 2. - 0.5 + grid
                       ) * self.downsample_ratios[i]
                pwh = (F.sigmoid(fg_pred[:, 2:4]) *
                       2)**2 * anch[i][idx] * self.downsample_ratios[i]
                pxywh = paddle.concat([pxy, pwh], -1)
                pxyxy = self.xywh2xyxy(pxywh)
                pxyxys.append(pxyxy)

            if empty_feats_num == 3:
                # print('batch_idx: {}, empty_feats_num == 3'.format(batch_idx))
                continue
            pxyxys = paddle.concat(pxyxys, 0)
            if pxyxys.shape[0] == 0:
                continue
            p_obj = paddle.concat(p_obj, 0)
            p_cls = paddle.concat(p_cls, 0)
            from_which_layer = paddle.concat(from_which_layer, 0)
            all_b = paddle.concat(all_b, 0)
            all_a = paddle.concat(all_a, 0)
            all_gj = paddle.concat(all_gj, 0)
            all_gi = paddle.concat(all_gi, 0)
            all_anch = paddle.concat(all_anch, 0)

            pair_wise_iou = self.box_iou(txyxy, pxyxys)  ###

            pair_wise_iou_loss = -paddle.log(pair_wise_iou + 1e-8)

            top_k, _ = paddle.topk(pair_wise_iou,
                                   min(10, pair_wise_iou.shape[1]),
                                   1)  # 28.37482071
            dynamic_ks = paddle.clip(
                paddle.cast(paddle.floor(top_k.sum(1)), 'int32'), min=1)

            gt_cls_per_image = (paddle.tile(
                F.one_hot(
                    paddle.cast(this_target[:, 1], 'int32'),
                    self.num_classes).unsqueeze(1), [1, pxyxys.shape[0], 1]))
            # [3, 48, 80] 144.

            num_gt = this_target.shape[0]
            cls_preds_ = (
                F.sigmoid(paddle.tile(p_cls.unsqueeze(0), [num_gt, 1, 1])) *
                F.sigmoid(paddle.tile(p_obj.unsqueeze(0), [num_gt, 1, 1])))
            # [3, 48, 80] 53.36702728

            y = cls_preds_.sqrt_()  # 83.95690918
            pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
                paddle.log(y / (1 - y)), gt_cls_per_image,
                reduction="none").sum(-1)
            # [3, 48] 671.70153809
            del cls_preds_

            cost = (pair_wise_cls_loss + 3.0 * pair_wise_iou_loss)
            '''
            matching_matrix = paddle.zeros_like(cost) # [3. 48]
            for gt_idx in range(num_gt):
                _, pos_idx = paddle.topk(
                    cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
                )
                matching_matrix[gt_idx, pos_idx] = 1.0 # shit bug, not [gt_idx][pos_idx]
            del top_k, dynamic_ks
            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                _, cost_argmin = paddle.min(cost[:, anchor_matching_gt > 1], 0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = paddle.cast((matching_matrix.sum(0) > 0.0) * 1.0, 'int64') # bool index bug
            matched_gt_inds = matching_matrix[fg_mask_inboxes].argmax(0)
            #matched_gt_inds = matching_matrix[paddle.tile(fg_mask_inboxes, [len(fg_mask_inboxes), 1])].argmax(0)
            paddle.masked_select(matching_matrix, paddle.tile(fg_mask_inboxes.unsqueeze(0), [len(matching_matrix), 1])))
            paddle.masked_select(matching_matrix, paddle.tile(fg_mask_inboxes.unsqueeze(0), [len(matching_matrix), 1])))
            paddle.gather(matching_matrix, fg_mask_inboxes.unsqueeze(0))
            '''
            matching_matrix = np.zeros(cost.shape)  # [3. 48]
            for gt_idx in range(num_gt):
                _, pos_idx = paddle.topk(
                    cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
                matching_matrix[
                    gt_idx, pos_idx] = 1.0  # shit bug, not [gt_idx][pos_idx]
            del top_k, dynamic_ks

            anchor_matching_gt = matching_matrix.sum(0)
            # print('batch_idx: no ifelse {}'.format(batch_idx), cost.shape, cost.sum(), (anchor_matching_gt > 1).sum())
            if (anchor_matching_gt > 1).sum() > 0:
                try:
                    # cost.shape [6, 101] # when batch_idx==7
                    cost_argmin = np.argmin(
                        cost.numpy()[:, anchor_matching_gt > 1], 0)
                    #print('batch_idx: {}'.format(batch_idx), cost.shape, cost.sum())
                except:
                    print('batch_idx: {}, cost_argmin bug'.format(batch_idx))
                    embed()

                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = matching_matrix.sum(0) > 0.0  # [48]
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(
                0)  # [27]

            from_which_layer = from_which_layer[fg_mask_inboxes]
            all_b = all_b[fg_mask_inboxes]
            all_a = all_a[fg_mask_inboxes]
            all_gj = all_gj[fg_mask_inboxes]
            all_gi = all_gi[fg_mask_inboxes]
            all_anch = all_anch[fg_mask_inboxes]  # [27, 2]

            this_target = this_target[matched_gt_inds]
            if len(this_target.shape) == 1:
                this_target = this_target.unsqueeze(0)

            for i in range(nl):
                layer_idx = from_which_layer == i
                if layer_idx.sum() == 0:  # single gpu ok, but multi-gpu may bug
                    #print('layer_idx.sum() == 0:')
                    #matching_targets[i].append([])
                    #embed()
                    continue
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])
                try:
                    matching_targets[i].append(this_target[layer_idx])
                    matching_anchs[i].append(all_anch[layer_idx])
                except:
                    print('layer_idx = from_which_layer == i')
                    embed()

        for i in range(nl):
            if matching_targets[i] != []:
                matching_bs[i] = paddle.concat(matching_bs[i], 0)
                matching_as[i] = paddle.concat(matching_as[i], 0)
                matching_gjs[i] = paddle.concat(matching_gjs[i], 0)
                matching_gis[i] = paddle.concat(matching_gis[i], 0)
                matching_targets[i] = paddle.concat(matching_targets[i], 0)
                matching_anchs[i] = paddle.concat(matching_anchs[i], 0)
            else:
                matching_bs[i] = paddle.to_tensor([], dtype='int64')
                matching_as[i] = paddle.to_tensor([], dtype='int64')
                matching_gjs[i] = paddle.to_tensor([], dtype='int64')
                matching_gis[i] = paddle.to_tensor([], dtype='int64')
                matching_targets[i] = paddle.to_tensor([], dtype='int64')
                matching_anchs[i] = paddle.to_tensor([], dtype='int64')

        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs

    def find_3_positive(self, p, targets, all_anchors):  # targets [143, 6]
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        indices, anch = [], []
        gain = paddle.ones([7])  # normalized to gridspace gain
        #gain = np.ones(7, dtype=np.float32)  # normalized to gridspace gain
        ai = paddle.tile(paddle.arange(na).reshape([na, 1]), [1, nt]) * 1.0
        # ai = np.tile(
        #     np.arange(
        #         na, dtype=np.float32).reshape(na, 1),
        #     [1, nt])  # same as .repeat_interleave(nt)
        targets = paddle.concat((paddle.tile(targets, [na, 1, 1]),
                                 ai.unsqueeze(-1)), 2)  # append anchor indices

        g = 0.5  # bias
        off = paddle.to_tensor(self.off)

        for i in range(len(p)):  # for i in range(self.nl):
            anchors = all_anchors[i] / self.downsample_ratios[i]
            gain[2:6] = paddle.to_tensor(
                p[i].shape, dtype=np.float32)[[3, 2, 3, 2]]  # xyxy gain
            # [1. , 1. , 80., 80., 80., 80., 1. ]

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio # [3, 22, 2]
                j = paddle.maximum(r, 1. / r).max(2) < self.anchor_t  # compare
                ### j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy # [3, 2, 7]
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = np.stack([np.ones_like(j), j, k, l, m])
                t = paddle.to_tensor(np.tile(t, [5, 1, 1])[j])  # (5, 23, 7)[j]
                ### todo
                offsets = (paddle.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].astype(np.int64).T  # image, class
            gxy = t[:, 2:4]  # grid xy
            # gwh = t[:, 4:6]  # grid wh no use
            gij = (gxy - offsets).astype(np.int64)
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].astype(np.int64)  # anchor indices
            indices.append((b, a, gj.clip(0, gain[3] - 1), gi.clip(
                0, gain[2] - 1)))  # image, anchor, grid indices
            anch.append(anchors[a])  # anchors

        return indices, anch
