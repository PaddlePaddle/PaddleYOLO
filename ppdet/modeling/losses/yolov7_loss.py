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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register
from ..bbox_utils import bbox_iou

__all__ = ['YOLOv7Loss']


@register
class YOLOv7Loss(nn.Layer):
    """
    this code is based on https://github.com/WongKinYiu/yolov7
    Note: Please use paddle 2.3.0+
    """
    __shared__ = ['num_classes', 'use_aux']

    def __init__(self,
                 num_classes=80,
                 use_aux=False,
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
        self.use_aux = use_aux
        self.balance = balance
        if self.use_aux:
            self.balance = balance * 2
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
        if self.use_aux:
            self.downsample_ratios = downsample_ratios * 2
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
        if self.use_aux:
            anchors = paddle.concat([anchors, anchors], 0)
        assert len(head_outs) == len(anchors)
        yolo_losses = dict()
        inputs = []
        for i, pi in enumerate(head_outs):
            bs, ch, h, w = pi.shape
            pi = pi.reshape((bs, self.na, ch // self.na, h, w)).transpose(
                (0, 1, 3, 4, 2))
            inputs.append(pi)

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
        targets = paddle.concat(
            [yolov7_gt_index, yolov7_gt_class, yolov7_gt_bbox], 1)

        lcls, lbox, lobj = paddle.zeros([1]), paddle.zeros([1]), paddle.zeros(
            [1])
        bs, as_, gjs, gis, targets, anchors = self.build_targets(
            inputs, targets, gt_targets['image'], anchors)
        pre_gen_gains = [
            paddle.to_tensor(pp.shape)[[3, 2, 3, 2]] for pp in inputs
        ]

        # Losses
        for i, pi in enumerate(inputs):
            b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]
            tobj = paddle.zeros_like(pi[..., 0])
            n = b.shape[0]  # number of targets
            if n:
                mask = paddle.stack([b, a, gj, gi], 1)
                ps = pi.gather_nd(mask)
                # Regression
                grid = paddle.stack([gi, gj], 1)
                if len(ps.shape) == 1:
                    ps = ps.unsqueeze(0)
                pxy = F.sigmoid(ps[:, :2]) * 2. - 0.5
                pwh = (F.sigmoid(ps[:, 2:4]) * 2)**2 * anchors[i]
                pbox = paddle.concat([pxy, pwh], 1)  # predicted box
                selected_tbox = targets[i][:, 2:6] * pre_gen_gains[i]
                selected_tbox[:, :2] -= grid
                iou = bbox_iou(
                    pbox.T,
                    selected_tbox.T,
                    x1y1x2y2=False,
                    ciou=True,
                    eps=1e-7)
                lbox += (1.0 - iou).mean()

                # Objectness
                score_iou = paddle.cast(iou.detach().clip(0), tobj.dtype)
                with paddle.no_grad():
                    x = paddle.gather_nd(tobj, mask)
                    tobj = paddle.scatter_nd_add(
                        tobj, mask, (1.0 - self.gr) + self.gr * score_iou - x)

                # Classification
                selected_tcls = paddle.cast(targets[i][:, 1], 'int64')
                if self.num_classes > 1:  # cls loss (only if multiple classes)
                    t = paddle.full_like(ps[:, 5:],
                                         self.cls_neg_label)  # targets
                    t[range(n), selected_tcls] = self.cls_pos_label
                    lcls += self.BCEcls(ps[:, 5:], t)

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
        # [x, y, w, h] to [x1, y1, x2, y2]
        y = x.clone()  # Tensor not numpy
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y

    def box_iou(self, box1, box2):
        """
        [N, 4] [M, 4] to get [N, M] ious, boxes in [x1, y1, x2, y2] format.
        """

        def box_area(box):
            return (box[2] - box[0]) * (box[3] - box[1])

        area1 = box_area(box1.T)
        area2 = box_area(box2.T)
        inter = (paddle.minimum(box1[:, None, 2:], box2[:, 2:]) -
                 paddle.maximum(box1[:, None, :2], box2[:, :2])).clip(0).prod(2)
        return inter / (area1[:, None] + area2 - inter)

    def build_targets(self, p, targets, imgs, anchors):
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
            txywh = this_target[:, 2:6] * imgs[batch_idx].shape[1]  # get 640
            txyxy = self.xywh2xyxy(txywh)

            pxyxys, p_cls, p_obj = [], [], []
            from_which_layer = []
            all_b, all_a, all_gj, all_gi = [], [], [], []
            all_anch = []

            empty_feats_num = 0
            for i, pi in enumerate(p):
                idx = (indices[i][0] == batch_idx)
                if idx.sum() == 0:
                    empty_feats_num += 1
                    continue
                b, a, gj, gi = indices[i][0][idx], indices[i][1][idx], indices[
                    i][2][idx], indices[i][3][idx]
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append(paddle.ones([len(b)]) * i)

                fg_pred = pi[b, a, gj, gi]
                if len(fg_pred.shape) == 1:  # paddle2.3 index
                    fg_pred = fg_pred.unsqueeze(0)

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

            if empty_feats_num == 3:  # note
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

            pair_wise_iou = self.box_iou(txyxy, pxyxys)
            # [N, 4] [M, 4] to get [N, M] ious

            pair_wise_iou_loss = -paddle.log(pair_wise_iou + 1e-8)

            top_k, _ = paddle.topk(pair_wise_iou,
                                   min(10, pair_wise_iou.shape[1]), 1)
            dynamic_ks = paddle.clip(
                paddle.cast(paddle.floor(top_k.sum(1)), 'int32'), min=1)

            gt_cls_per_image = (paddle.tile(
                F.one_hot(
                    paddle.cast(this_target[:, 1], 'int32'),
                    self.num_classes).unsqueeze(1), [1, pxyxys.shape[0], 1]))

            num_gt = this_target.shape[0]
            cls_preds_ = (
                F.sigmoid(paddle.tile(p_cls.unsqueeze(0), [num_gt, 1, 1])) *
                F.sigmoid(paddle.tile(p_obj.unsqueeze(0), [num_gt, 1, 1])))

            y = cls_preds_.sqrt_()
            pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
                paddle.log(y / (1 - y)), gt_cls_per_image,
                reduction="none").sum(-1)
            del cls_preds_

            cost = (pair_wise_cls_loss + 3.0 * pair_wise_iou_loss)

            matching_matrix = np.zeros(cost.shape)  # [3. 48]
            for gt_idx in range(num_gt):
                _, pos_idx = paddle.topk(
                    cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
                matching_matrix[gt_idx, pos_idx] = 1.0
                # paddle2.3 index. not [gt_idx][pos_idx], diff with torch
            del top_k, dynamic_ks

            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                cost_argmin = np.argmin(cost.numpy()[:, anchor_matching_gt > 1],
                                        0)

                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = matching_matrix.sum(0) > 0.0
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)

            from_which_layer = from_which_layer[fg_mask_inboxes]
            all_b = all_b[fg_mask_inboxes]
            all_a = all_a[fg_mask_inboxes]
            all_gj = all_gj[fg_mask_inboxes]
            all_gi = all_gi[fg_mask_inboxes]
            all_anch = all_anch[fg_mask_inboxes]

            this_target = this_target[matched_gt_inds]
            if len(this_target.shape) == 1:
                this_target = this_target.unsqueeze(0)

            for i in range(nl):
                layer_idx = from_which_layer == i
                if layer_idx.sum() == 0:  # note
                    continue
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])

                # note: be careful
                matching_targets[i].append(this_target[layer_idx])
                matching_anchs[i].append(all_anch[layer_idx])

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

    def find_3_positive(self, p, targets, all_anchors):
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        indices, anch = [], []
        gain = paddle.ones([7])  # normalized to gridspace gain
        ai = paddle.tile(paddle.arange(na).reshape([na, 1]), [1, nt]) * 1.0
        targets = paddle.concat((paddle.tile(targets, [na, 1, 1]),
                                 ai.unsqueeze(-1)), 2)  # append anchor indices

        g = 0.5  # bias
        off = paddle.to_tensor(self.off)

        for i in range(len(p)):
            anchors = all_anchors[i] / self.downsample_ratios[i]
            gain[2:6] = paddle.to_tensor(
                p[i].shape, dtype=np.float32)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = paddle.maximum(r, 1. / r).max(2) < self.anchor_t  # compare
                # j = torch.max(r, 1. / r).max(2)[0]
                t = t[j]  # filter

                if t.shape[0] == 0:
                    t = targets[0]
                    offsets = 0
                else:
                    # Offsets
                    gxy = t[:, 2:4]  # grid xy
                    gxi = gain[[2, 3]] - gxy  # inverse
                    j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                    l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                    j = np.stack([np.ones_like(j), j, k, l, m])
                    t = paddle.to_tensor(np.tile(t, [5, 1, 1])[j])
                    offsets = (paddle.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].astype(np.int64).T
            gxy = t[:, 2:4]  # grid xy
            # gwh = t[:, 4:6]  # grid wh no use
            gij = (gxy - offsets).astype(np.int64)
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].astype(np.int64)  # anchor indices
            indices.append((b, a, gj.clip(0, gain[3] - 1),
                            gi.clip(0, gain[2] - 1)))
            # bs, anchor, gj, gi
            anch_ = anchors[a]
            if len(anch_.shape) == 1:
                anch_ = anch_.unsqueeze(0)
            anch.append(anch_)

        return indices, anch
