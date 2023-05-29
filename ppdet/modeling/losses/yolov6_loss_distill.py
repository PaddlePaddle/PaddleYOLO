import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppdet.modeling.assigners.task_aligned_assigner import TaskAlignedAssigner
from ppdet.modeling.assigners.atss_assigner import ATSSAssigner
from ppdet.modeling.ops import get_static_shape
from ..assigners.utils import generate_anchors_for_grid_cell
from ..bbox_utils import batch_distance2bbox
from ..losses import GIoULoss, SIoULoss


class loss_distill:
    # add reg_preds_lrtb
    __shared__ = [
        'num_classes', 'eval_size', 'trt', 'exclude_nms',
        'exclude_post_process', 'self_distill'
    ]
    __inject__ = ['static_assigner', 'assigner', 'nms']

    def __init__(
            self,
            num_classes=80,
            fpn_strides=[8, 16, 32],
            grid_cell_scale=5.0,
            grid_cell_offset=0.5,
            reg_max=16,  # reg_max=0 if use_dfl is False
            use_dfl=True,  # False in n/s version, True in m/l version
            static_assigner_epoch=4,  # warmup_epoch
            iou_type='giou',  # 'siou' in n version
            loss_weight={
                'cls': 1.0,
                'iou': 2.5,
                'dfl': 0.5,  # used in m/l version
                'cwd': 10.0,  # used when self_distill=True, in m/l version
            },
            distill_feat=False,
            distill_weight={
                'cls': 1.0,
                'dfl': 1.0,
            },
            print_l1_loss=True):
        super(loss_distill, self).__init__()
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.reg_max = reg_max
        self.use_dfl = use_dfl

        self.static_assigner_epoch = static_assigner_epoch
        self.warmup_assigner = ATSSAssigner(9, num_classes=self.num_classes)
        self.assigner_task = TaskAlignedAssigner(topk=13, num_classes=self.num_classes, alpha=1.0, beta=6.0)
        self.print_l1_loss = print_l1_loss
        self.varifocal_loss = VarifocalLoss()
        self.bbox_loss = BboxLoss(self.reg_max, self.use_dfl, iou_type)
        self.proj = paddle.linspace(0, self.reg_max, self.reg_max + 1)
        # for self-distillation
        self.loss_weight = loss_weight
        self.distill_feat = distill_feat
        self.distill_weight = distill_weight

    def __call__(self, head_outs, t_outputs, epoch_num, max_epoch, temperature):
        feats, pred_scores, pred_distri, _, gt_meta = head_outs

        anchors, anchor_points, num_anchors_list, stride_tensor = \
            generate_anchors_for_grid_cell(
                feats, self.fpn_strides, self.grid_cell_scale,
                self.grid_cell_offset)
        t_feats, t_pred_scores, t_pred_distri = t_outputs[0], t_outputs[-2], t_outputs[-1]

        t_anchors,t_anchor_points, t_num_anchors_list, t_stride_tensor, = \
            generate_anchors_for_grid_cell(
                t_feats, self.fpn_strides, self.grid_cell_scale,
                self.grid_cell_offset)

        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self._bbox_decode(anchor_points_s, pred_distri)

        t_anchor_points_s = t_anchor_points / t_stride_tensor
        t_pred_bboxes = self._bbox_decode(t_anchor_points_s, t_pred_distri)

        gt_labels = gt_meta['gt_class']
        gt_bboxes = gt_meta['gt_bbox']
        pad_gt_mask = gt_meta['pad_gt_mask']
        # label assignment
        if gt_meta['epoch_id'] < self.static_assigner_epoch:
            assigned_labels, assigned_bboxes, assigned_scores, fg_mask = \
                self.warmup_assigner(
                    anchors,
                    num_anchors_list,
                    gt_labels,
                    gt_bboxes,
                    pad_gt_mask,
                    bg_index=self.num_classes,
                    pred_bboxes=pred_bboxes.detach() * stride_tensor)
        else:
            assigned_labels, assigned_bboxes, assigned_scores, fg_mask = \
                self.assigner_task(
                    pred_scores.detach(),
                    pred_bboxes.detach() * stride_tensor,
                    anchor_points,
                    num_anchors_list,
                    gt_labels,
                    gt_bboxes,
                    pad_gt_mask,
                    bg_index=self.num_classes)
        # rescale bbox
        assigned_bboxes /= stride_tensor

        # cls loss: varifocal_loss
        assigned_labels = paddle.where(fg_mask > 0, assigned_labels,
                                       paddle.full_like(assigned_labels, self.num_classes))
        one_hot_label = F.one_hot(assigned_labels,
                                  self.num_classes + 1)[..., :-1]
        loss_cls = self.varifocal_loss(pred_scores, assigned_scores,
                                       one_hot_label)
        assigned_scores_sum = assigned_scores.sum()
        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.all_reduce(assigned_scores_sum)
            assigned_scores_sum = paddle.clip(
                assigned_scores_sum / paddle.distributed.get_world_size(),
                min=1)
        loss_cls /= assigned_scores_sum

        # bbox loss
        loss_iou, loss_dfl, d_loss_dfl, loss_l1 = \
            self.bbox_loss(pred_distri,
                           pred_bboxes,
                           t_pred_distri,
                           t_pred_bboxes,
                           temperature,
                           anchor_points_s,
                           assigned_bboxes,
                           assigned_scores,
                           assigned_scores_sum,
                           fg_mask)
        logits_student = pred_scores
        logits_teacher = t_pred_scores
        distill_num_classes = self.num_classes
        d_loss_cls = self.distill_loss_cls(logits_student, logits_teacher, distill_num_classes, temperature)
        if self.distill_feat:
            d_loss_cw = self.distill_loss_cw(feats, t_feats)
        else:
            d_loss_cw = paddle.to_tensor(0.)
        import math
        distill_weightdecay = ((1 - math.cos(epoch_num * math.pi / max_epoch)) / 2) * (0.01 - 1) + 1
        d_loss_dfl *= distill_weightdecay
        d_loss_cls *= distill_weightdecay
        d_loss_cw *= distill_weightdecay

        loss_cls_all = loss_cls + d_loss_cls * self.distill_weight['cls']
        loss_dfl_all = loss_dfl + d_loss_dfl * self.distill_weight['dfl']

        loss = self.loss_weight['cls'] * loss_cls_all + \
               self.loss_weight['iou'] * loss_iou + \
               self.loss_weight['dfl'] * loss_dfl_all + \
               self.loss_weight['cwd'] * d_loss_cw
        num_gpus = gt_meta.get('num_gpus', 8)
        out_dict = {
            'loss': loss * num_gpus,
            'loss_cls': self.loss_weight['cls'] * loss_cls_all,
            'loss_iou': self.loss_weight['iou'] * loss_iou,
            'loss_dfl': self.loss_weight['dfl'] * loss_dfl_all,
            'loss_cwd': self.loss_weight['cwd'] * d_loss_cw
        }

        if self.print_l1_loss:
            # just see convergence
            out_dict.update({'loss_l1': loss_l1})
        return out_dict

    def _bbox_decode(self, anchor_points, pred_dist):        
        ### diff with PPYOLOEHead
        if self.use_dfl:
            b, l, _ = get_static_shape(pred_dist)
            pred_dist = F.softmax(
                pred_dist.reshape([b, l, 4, self.reg_max + 1])).matmul(self.proj.reshape((-1, 1))).squeeze()

        return batch_distance2bbox(anchor_points, pred_dist)

    def distill_loss_cls(self, logits_student, logits_teacher, num_classes, temperature=20):
        logits_student = logits_student.reshape([-1, num_classes])
        logits_teacher = logits_teacher.reshape([-1, num_classes])
        pred_student = F.softmax(logits_student / temperature, axis=1)
        pred_teacher = F.softmax(logits_teacher / temperature, axis=1)
        log_pred_student = paddle.log(pred_student)

        d_loss_cls = F.kl_div(log_pred_student, pred_teacher, reduction="sum")
        d_loss_cls *= temperature ** 2
        return d_loss_cls

    def distill_loss_cw(self, s_feats, t_feats, temperature=1):
        N, C, H, W = s_feats[0].shape
        # print(N,C,H,W)
        loss_cw = F.kl_div(F.log_softmax(s_feats[0].reshape([N, C, H * W]) / temperature, axis=2),
                           F.log_softmax(t_feats[0].reshape([N, C, H * W]).detach() / temperature, axis=2),
                           reduction='sum') * (temperature * temperature) / (N * C)

        N, C, H, W = s_feats[1].shape
        # print(N,C,H,W)
        loss_cw += F.kl_div(F.log_softmax(s_feats[1].reshape([N, C, H * W]) / temperature, axis=2),
                            F.log_softmax(t_feats[1].reshape([N, C, H * W]).detach() / temperature, axis=2),
                            reduction='sum') * (temperature * temperature) / (N * C)

        N, C, H, W = s_feats[2].shape
        # print(N,C,H,W)
        loss_cw += F.kl_div(F.log_softmax(s_feats[2].rehape([N, C, H * W]) / temperature, axis=2),
                            F.log_softmax(t_feats[2].reshape([N, C, H * W]).detach() / temperature, axis=2),
                            reduction='sum') * (temperature * temperature) / (N * C)
        # print(loss_cw)
        return loss_cw


class VarifocalLoss(nn.Layer):
    def __init__(self):
        super(VarifocalLoss, self).__init__()

    def forward(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
        loss = F.binary_cross_entropy(
            pred_score, gt_score, weight=weight, reduction='sum')
        return loss


class BboxLoss(nn.Layer):
    def __init__(self, reg_max, use_dfl=False, iou_type='giou'):
        super(BboxLoss, self).__init__()
        assert iou_type in ['giou', 'siou'], "only support giou and siou loss."
        if iou_type == 'siou':
            self.iou_loss = SIoULoss()
        else:
            self.iou_loss = GIoULoss()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, t_pred_dist, t_pred_bboxes, temperature, anchor_points,
                target_bboxes, target_scores, target_scores_sum, fg_mask):
        # select positive samples mask
        num_pos = fg_mask.sum()
        if num_pos > 0:
            # iou loss
            bbox_mask = fg_mask.unsqueeze(-1).tile([1, 1, 4])
            pred_bboxes_pos = paddle.masked_select(pred_bboxes,
                                                   bbox_mask).reshape([-1, 4])

            target_bboxes_pos = paddle.masked_select(
                target_bboxes, bbox_mask).reshape([-1, 4])

            bbox_weight = paddle.masked_select(
                target_scores.sum(-1), fg_mask).unsqueeze(-1)
            loss_iou = self.iou_loss(pred_bboxes_pos,
                                     target_bboxes_pos) * bbox_weight

            # l1 loss just see the convergence, same in PPYOLOEHead
            loss_l1 = F.l1_loss(pred_bboxes_pos, target_bboxes_pos)

            if target_scores_sum == 0:
                loss_iou = loss_iou.sum()
            else:
                loss_iou = loss_iou.sum() / target_scores_sum

            # dfl loss
            if self.use_dfl:
                dist_mask = fg_mask.unsqueeze(-1).tile(
                    [1, 1, (self.reg_max + 1) * 4])
                pred_dist_pos = paddle.masked_select(
                    pred_dist, dist_mask).reshape([-1, 4, self.reg_max + 1])
                t_pred_dist_pos = paddle.masked_select(
                    t_pred_dist, dist_mask).reshape([-1, 4, self.reg_max + 1])
                target_ltrb = self._bbox2distance(anchor_points, target_bboxes)
                target_ltrb_pos = paddle.masked_select(
                    target_ltrb, bbox_mask).reshape([-1, 4])
                loss_dfl = self._df_loss(pred_dist_pos,
                                         target_ltrb_pos) * bbox_weight
                d_loss_dfl = self.distill_loss_dfl(pred_dist_pos, t_pred_dist_pos, temperature) * bbox_weight
                if target_scores_sum == 0:
                    loss_dfl = loss_dfl.sum()
                    d_loss_dfl = d_loss_dfl.sum()
                else:
                    loss_dfl = loss_dfl.sum() / target_scores_sum
                    d_loss_dfl = d_loss_dfl.sum() / target_scores_sum
            else:
                loss_dfl = pred_dist.sum() * 0.
                d_loss_dfl = pred_dist.sum() * 0.

        else:

            loss_iou = pred_dist.sum() * 0.
            loss_dfl = pred_dist.sum() * 0.
            d_loss_dfl = pred_dist.sum() * 0.
            loss_l1 = 0

        return loss_iou, loss_dfl, d_loss_dfl, loss_l1

    def _bbox2distance(self, points, bbox):
        x1y1, x2y2 = paddle.split(bbox, 2, -1)
        lt = points - x1y1
        rb = x2y2 - points
        return paddle.concat([lt, rb], -1).clip(0, self.reg_max - 0.01)

    def _df_loss(self, pred_dist, target):
        target_left = paddle.cast(target, 'int64')
        target_right = target_left + 1
        weight_left = target_right.astype('float32') - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(
            pred_dist, target_left, reduction='none') * weight_left
        loss_right = F.cross_entropy(
            pred_dist, target_right, reduction='none') * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)

    def distill_loss_dfl(self, logits_student, logits_teacher, temperature=20):

        logits_student = logits_student.reshape([-1, 17])
        logits_teacher = logits_teacher.reshape([-1, 17])
        pred_student = F.softmax(logits_student / temperature, axis=1)
        pred_teacher = F.softmax(logits_teacher / temperature, axis=1)
        log_pred_student = paddle.log(pred_student)

        d_loss_dfl = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
        d_loss_dfl *= temperature ** 2
        return d_loss_dfl
