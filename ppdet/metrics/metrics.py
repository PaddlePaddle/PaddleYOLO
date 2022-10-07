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

import os
import sys
import json
import paddle
import numpy as np
import typing
from collections import defaultdict
from pathlib import Path

from .map_utils import prune_zero_padding, DetectionMAP
from .coco_utils import get_infer_results, cocoapi_eval
from ppdet.data.source.category import get_categories

from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

__all__ = ['Metric', 'COCOMetric', 'VOCMetric', 'get_infer_results']


class Metric(paddle.metric.Metric):
    def name(self):
        return self.__class__.__name__

    def reset(self):
        pass

    def accumulate(self):
        pass

    # paddle.metric.Metric defined :metch:`update`, :meth:`accumulate`
    # :metch:`reset`, in ppdet, we also need following 2 methods:

    # abstract method for logging metric results
    def log(self):
        pass

    # abstract method for getting metric results
    def get_results(self):
        pass


class COCOMetric(Metric):
    def __init__(self, anno_file, **kwargs):
        self.anno_file = anno_file
        self.clsid2catid = kwargs.get('clsid2catid', None)
        if self.clsid2catid is None:
            self.clsid2catid, _ = get_categories('COCO', anno_file)
        self.classwise = kwargs.get('classwise', False)
        self.output_eval = kwargs.get('output_eval', None)
        # TODO: bias should be unified
        self.bias = kwargs.get('bias', 0)
        self.save_prediction_only = kwargs.get('save_prediction_only', False)
        self.iou_type = kwargs.get('IouType', 'bbox')

        if not self.save_prediction_only:
            assert os.path.isfile(anno_file), \
                    "anno_file {} not a file".format(anno_file)

        if self.output_eval is not None:
            Path(self.output_eval).mkdir(exist_ok=True)

        self.reset()

    def reset(self):
        # only bbox and mask evaluation support currently
        self.results = {'bbox': [], 'mask': [], 'segm': [], 'keypoint': []}
        self.eval_results = {}

    def update(self, inputs, outputs):
        outs = {}
        # outputs Tensor -> numpy.ndarray
        for k, v in outputs.items():
            outs[k] = v.numpy() if isinstance(v, paddle.Tensor) else v

        # multi-scale inputs: all inputs have same im_id
        if isinstance(inputs, typing.Sequence):
            im_id = inputs[0]['im_id']
        else:
            im_id = inputs['im_id']
        outs['im_id'] = im_id.numpy() if isinstance(im_id,
                                                    paddle.Tensor) else im_id

        infer_results = get_infer_results(
            outs, self.clsid2catid, bias=self.bias)
        self.results['bbox'] += infer_results[
            'bbox'] if 'bbox' in infer_results else []
        self.results['mask'] += infer_results[
            'mask'] if 'mask' in infer_results else []
        self.results['segm'] += infer_results[
            'segm'] if 'segm' in infer_results else []
        self.results['keypoint'] += infer_results[
            'keypoint'] if 'keypoint' in infer_results else []

    def accumulate(self):
        if len(self.results['bbox']) > 0:
            output = "bbox.json"
            if self.output_eval:
                output = os.path.join(self.output_eval, output)
            with open(output, 'w') as f:
                json.dump(self.results['bbox'], f)
                logger.info('The bbox result is saved to bbox.json.')

            if self.save_prediction_only:
                logger.info('The bbox result is saved to {} and do not '
                            'evaluate the mAP.'.format(output))
            else:
                bbox_stats = cocoapi_eval(
                    output,
                    'bbox',
                    anno_file=self.anno_file,
                    classwise=self.classwise)
                self.eval_results['bbox'] = bbox_stats
                sys.stdout.flush()

        if len(self.results['mask']) > 0:
            output = "mask.json"
            if self.output_eval:
                output = os.path.join(self.output_eval, output)
            with open(output, 'w') as f:
                json.dump(self.results['mask'], f)
                logger.info('The mask result is saved to mask.json.')

            if self.save_prediction_only:
                logger.info('The mask result is saved to {} and do not '
                            'evaluate the mAP.'.format(output))
            else:
                seg_stats = cocoapi_eval(
                    output,
                    'segm',
                    anno_file=self.anno_file,
                    classwise=self.classwise)
                self.eval_results['mask'] = seg_stats
                sys.stdout.flush()

        if len(self.results['segm']) > 0:
            output = "segm.json"
            if self.output_eval:
                output = os.path.join(self.output_eval, output)
            with open(output, 'w') as f:
                json.dump(self.results['segm'], f)
                logger.info('The segm result is saved to segm.json.')

            if self.save_prediction_only:
                logger.info('The segm result is saved to {} and do not '
                            'evaluate the mAP.'.format(output))
            else:
                seg_stats = cocoapi_eval(
                    output,
                    'segm',
                    anno_file=self.anno_file,
                    classwise=self.classwise)
                self.eval_results['mask'] = seg_stats
                sys.stdout.flush()

    def log(self):
        pass

    def get_results(self):
        return self.eval_results


class VOCMetric(Metric):
    def __init__(self,
                 label_list,
                 class_num=20,
                 overlap_thresh=0.5,
                 map_type='11point',
                 is_bbox_normalized=False,
                 evaluate_difficult=False,
                 classwise=False,
                 output_eval=None,
                 save_prediction_only=False):
        assert os.path.isfile(label_list), \
                "label_list {} not a file".format(label_list)
        self.clsid2catid, self.catid2name = get_categories('VOC', label_list)

        self.overlap_thresh = overlap_thresh
        self.map_type = map_type
        self.evaluate_difficult = evaluate_difficult
        self.output_eval = output_eval
        self.save_prediction_only = save_prediction_only
        self.detection_map = DetectionMAP(
            class_num=class_num,
            overlap_thresh=overlap_thresh,
            map_type=map_type,
            is_bbox_normalized=is_bbox_normalized,
            evaluate_difficult=evaluate_difficult,
            catid2name=self.catid2name,
            classwise=classwise)

        self.reset()

    def reset(self):
        self.results = {'bbox': [], 'score': [], 'label': []}
        self.detection_map.reset()

    def update(self, inputs, outputs):
        bbox_np = outputs['bbox'].numpy() if isinstance(
            outputs['bbox'], paddle.Tensor) else outputs['bbox']
        bboxes = bbox_np[:, 2:]
        scores = bbox_np[:, 1]
        labels = bbox_np[:, 0]
        bbox_lengths = outputs['bbox_num'].numpy() if isinstance(
            outputs['bbox_num'], paddle.Tensor) else outputs['bbox_num']

        self.results['bbox'].append(bboxes.tolist())
        self.results['score'].append(scores.tolist())
        self.results['label'].append(labels.tolist())

        if bboxes.shape == (1, 1) or bboxes is None:
            return
        if self.save_prediction_only:
            return

        gt_boxes = inputs['gt_bbox']
        gt_labels = inputs['gt_class']
        difficults = inputs['difficult'] if not self.evaluate_difficult \
                            else None

        if 'scale_factor' in inputs:
            scale_factor = inputs['scale_factor'].numpy() if isinstance(
                inputs['scale_factor'],
                paddle.Tensor) else inputs['scale_factor']
        else:
            scale_factor = np.ones((gt_boxes.shape[0], 2)).astype('float32')

        bbox_idx = 0
        for i in range(len(gt_boxes)):
            gt_box = gt_boxes[i].numpy() if isinstance(
                gt_boxes[i], paddle.Tensor) else gt_boxes[i]
            h, w = scale_factor[i]
            gt_box = gt_box / np.array([w, h, w, h])
            gt_label = gt_labels[i].numpy() if isinstance(
                gt_labels[i], paddle.Tensor) else gt_labels[i]
            if difficults is not None:
                difficult = difficults[i].numpy() if isinstance(
                    difficults[i], paddle.Tensor) else difficults[i]
            else:
                difficult = None
            bbox_num = bbox_lengths[i]
            bbox = bboxes[bbox_idx:bbox_idx + bbox_num]
            score = scores[bbox_idx:bbox_idx + bbox_num]
            label = labels[bbox_idx:bbox_idx + bbox_num]
            gt_box, gt_label, difficult = prune_zero_padding(gt_box, gt_label,
                                                             difficult)
            self.detection_map.update(bbox, score, label, gt_box, gt_label,
                                      difficult)
            bbox_idx += bbox_num

    def accumulate(self):
        output = "bbox.json"
        if self.output_eval:
            output = os.path.join(self.output_eval, output)
            with open(output, 'w') as f:
                json.dump(self.results, f)
                logger.info('The bbox result is saved to bbox.json.')
        if self.save_prediction_only:
            return

        logger.info("Accumulating evaluatation results...")
        self.detection_map.accumulate()

    def log(self):
        map_stat = 100. * self.detection_map.get_map()
        logger.info("mAP({:.2f}, {}) = {:.2f}%".format(self.overlap_thresh,
                                                       self.map_type, map_stat))

    def get_results(self):
        return {'bbox': [self.detection_map.get_map()]}
