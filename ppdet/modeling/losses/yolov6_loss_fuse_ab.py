import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppdet.modeling.assigners.task_aligned_assigner import TaskAlignedAssigner
from ..assigners.utils import generate_anchors_for_grid_cell
from ..losses import GIoULoss, SIoULoss


class loss_fuse_ab:
    '''Loss computation func.'''

    def __init__(self,
                 fpn_strides=[8, 16, 32],
                 grid_cell_scale=5.0,
                 grid_cell_offset=0.5,
                 num_classes=80,
                 static_assigner_epoch=0,
                 use_dfl=True,
                 reg_max=16,
                 iou_type='giou',
                 loss_weight={
                     'cls': 1.0,
                     'iou': 2.5,
                     'dfl': 0.5},
                 print_l1_loss=True
                 ):
        super(loss_fuse_ab, self).__init__()
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
        self.assigner_ab = TaskAlignedAssigner(topk=26, alpha=1.0, beta=6.0)
        self.print_l1_loss = print_l1_loss
        self.varifocal_loss = VarifocalLoss()
        self.proj = paddle.linspace(0, self.reg_max, self.reg_max + 1)
        self.loss_weight = loss_weight

    def __call__(self, head_outs, gt_meta):
        feats, pred_scores, pred_distri = head_outs
        anchors, anchor_points, num_anchors_list, stride_tensor = \
            generate_anchors_for_grid_cell(
                feats, self.fpn_strides, self.grid_cell_scale,
                self.grid_cell_offset, mode='ab')

        anchor_points_s = anchor_points / stride_tensor
        pred_distri[..., :2] += anchor_points_s
        pred_bboxes = self.xywh2xyxy(pred_distri)

        gt_labels = gt_meta['gt_class']
        gt_bboxes = gt_meta['gt_bbox']
        pad_gt_mask = gt_meta['pad_gt_mask']
        # label assignment
        assigned_labels, assigned_bboxes, assigned_scores, fg_mask = \
            self.assigner_ab(
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
                           assigned_bboxes, assigned_scores,
                           assigned_scores_sum, fg_mask)

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

    def xywh2xyxy(self, bboxes):
        '''Transform bbox(xywh) to box(xyxy).'''
        bboxes[..., 0] = bboxes[..., 0] - bboxes[..., 2] * 0.5
        bboxes[..., 1] = bboxes[..., 1] - bboxes[..., 3] * 0.5
        bboxes[..., 2] = bboxes[..., 0] + bboxes[..., 2]
        bboxes[..., 3] = bboxes[..., 1] + bboxes[..., 3]
        return bboxes
    
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

    def froward(self, pred_dist, pred_bboxes, anchor_points,
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
