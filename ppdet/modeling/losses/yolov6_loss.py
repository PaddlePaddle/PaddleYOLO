import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppdet.modeling.assigners.task_aligned_assigner import TaskAlignedAssigner
from ppdet.modeling.assigners.atss_assigner import ATSSAssigner
from ..assigners.utils import generate_anchors_for_grid_cell
from ppdet.modeling.ops import get_static_shape
from ..bbox_utils import batch_distance2bbox
from ..losses import GIoULoss, SIoULoss


class ComputeLoss:
    '''Loss computation func.'''

    def __init__(self,
                 fpn_strides=[8, 16, 32],
                 grid_cell_scale=5.0,
                 grid_cell_offset=0.5,
                 num_classes=80,
                 static_assigner_epoch=4,
                 use_dfl=True,
                 reg_max=16,
                 iou_type='giou',
                 loss_weight={
                     'cls': 1.0,
                     'iou': 2.5,
                     'dfl': 0.5},
                 print_l1_loss=True
                 ):
        super(ComputeLoss, self).__init__()
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        assert iou_type in ['giou', 'siou'], "only support giou and siou loss."
        if iou_type == 'siou':
            self.iou_loss = SIoULoss()
        else:
            self.iou_loss = GIoULoss()
        self.reg_max = reg_max
        self.use_dfl = use_dfl
        self.static_assigner_epoch = static_assigner_epoch
        self.warmup_assigner = ATSSAssigner(9, num_classes=self.num_classes)
        self.formal_assigner = TaskAlignedAssigner(topk=13, alpha=1.0, beta=6.0)

        self.print_l1_loss = print_l1_loss
        self.varifocal_loss = VarifocalLoss()
        self.proj = paddle.linspace(0, self.reg_max, self.reg_max + 1)
        self.loss_weight = loss_weight

    def __call__(self, head_outs, gt_meta):
        feats, pred_scores, pred_distri = head_outs
        anchors, anchor_points, num_anchors_list, stride_tensor = \
            generate_anchors_for_grid_cell(
                feats, self.fpn_strides, self.grid_cell_scale,
                self.grid_cell_offset)

        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self._bbox_decode(anchor_points_s, pred_distri)

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
                self.formal_assigner(
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
        loss_l1, loss_iou, loss_dfl = \
            self._bbox_loss(pred_distri, pred_bboxes, anchor_points_s,
                            assigned_bboxes, assigned_scores,assigned_scores_sum, fg_mask)

        if self.use_dfl:
            loss = self.loss_weight['cls'] * loss_cls + \
                   self.loss_weight['iou'] * loss_iou + \
                   self.loss_weight['dfl'] * loss_dfl
            num_gpus = gt_meta.get('num_gpus', 8)
            out_dict = {
                'loss': loss * num_gpus,
                'loss_cls': self.loss_weight['cls'] * loss_cls,
                'loss_iou': self.loss_weight['iou'] * loss_iou,
                'loss_dfl': self.loss_weight['dfl'] * loss_dfl,
            }
        else:
            loss = self.loss_weight['cls'] * loss_cls + \
                   self.loss_weight['iou'] * loss_iou
            num_gpus = gt_meta.get('num_gpus', 8)
            out_dict = {
                'loss': loss * num_gpus,
                'loss_cls': self.loss_weight['cls'] * loss_cls,
                'loss_iou': self.loss_weight['iou'] * loss_iou,
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
            print("pred_dist",pred_dist.shape)
        return batch_distance2bbox(anchor_points, pred_dist)
    
    def _bbox_loss(self, pred_dist, pred_bboxes, anchor_points,
                assigned_bboxes, assigned_scores, assigned_scores_sum, fg_mask):
        # select positive samples mask
        mask_positive = fg_mask
        num_pos = mask_positive.sum()
        # pos/neg loss
        if num_pos > 0:
            # iou loss
            bbox_mask = mask_positive.unsqueeze(-1).tile([1, 1, 4])
            pred_bboxes_pos = paddle.masked_select(pred_bboxes,
                                                   bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = paddle.masked_select(
                assigned_bboxes, bbox_mask).reshape([-1, 4])
            bbox_weight = paddle.masked_select(
                assigned_scores.sum(-1), mask_positive).unsqueeze(-1)
            loss_iou = self.iou_loss(pred_bboxes_pos,
                                     assigned_bboxes_pos) * bbox_weight
            loss_iou = loss_iou.sum() / assigned_scores_sum

            # l1 loss just see the convergence, same in PPYOLOEHead
            loss_l1 = F.l1_loss(pred_bboxes_pos, assigned_bboxes_pos)

            # dfl loss ### diff with PPYOLOEHead
            if self.use_dfl:
                dist_mask = mask_positive.unsqueeze(-1).tile(
                    [1, 1, (self.reg_max + 1) * 4])
                pred_dist_pos = paddle.masked_select(
                    pred_dist, dist_mask).reshape([-1, 4, self.reg_max + 1])
                assigned_ltrb = self._bbox2distance(anchor_points,
                                                    assigned_bboxes)
                assigned_ltrb_pos = paddle.masked_select(
                    assigned_ltrb, bbox_mask).reshape([-1, 4])
                loss_dfl = self._df_loss(pred_dist_pos,
                                         assigned_ltrb_pos) * bbox_weight
                loss_dfl = loss_dfl.sum() / assigned_scores_sum
            else:
                loss_dfl = pred_dist.sum() * 0.
        else:
            loss_l1 = paddle.zeros([1])
            loss_iou = paddle.zeros([1])
            loss_dfl = pred_dist.sum() * 0.
        return loss_l1, loss_iou, loss_dfl

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


class VarifocalLoss(nn.Layer):
    def __init__(self):
        super(VarifocalLoss, self).__init__()

    def forward(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
        loss = F.binary_cross_entropy(
            pred_score, gt_score, weight=weight, reduction='sum')
        return loss

