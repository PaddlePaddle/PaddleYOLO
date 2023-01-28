# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import unicode_literals

import numpy as np
from PIL import Image, ImageDraw
import cv2
import math

from .colormap import colormap
from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

__all__ = ['visualize_results']


def visualize_results(image,
                      bbox_res,
                      mask_res,
                      segm_res,
                      keypoint_res,
                      pose3d_res,
                      im_id,
                      catid2name,
                      threshold=0.5):
    """
    Visualize bbox and mask results
    """
    if bbox_res is not None:
        image = draw_bbox(image, im_id, catid2name, bbox_res, threshold)
    return image


def draw_bbox(image, im_id, catid2name, bboxes, threshold):
    """
    Draw bbox on image
    """
    draw = ImageDraw.Draw(image)

    catid2color = {}
    color_list = colormap(rgb=True)[:40]
    for dt in np.array(bboxes):
        if im_id != dt['image_id']:
            continue
        catid, bbox, score = dt['category_id'], dt['bbox'], dt['score']
        if score < threshold:
            continue

        if catid not in catid2color:
            idx = np.random.randint(len(color_list))
            catid2color[catid] = color_list[idx]
        color = tuple(catid2color[catid])

        # draw bbox
        if len(bbox) == 4:
            # draw bbox
            xmin, ymin, w, h = bbox
            xmax = xmin + w
            ymax = ymin + h
            draw.line(
                [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
                 (xmin, ymin)],
                width=2,
                fill=color)
        elif len(bbox) == 8:
            x1, y1, x2, y2, x3, y3, x4, y4 = bbox
            draw.line(
                [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)],
                width=2,
                fill=color)
            xmin = min(x1, x2, x3, x4)
            ymin = min(y1, y2, y3, y4)
        else:
            logger.error('the shape of bbox must be [M, 4] or [M, 8]!')

        # draw label
        text = "{} {:.2f}".format(catid2name[catid], score)
        tw, th = draw.textsize(text)
        draw.rectangle(
            [(xmin + 1, ymin - th), (xmin + tw + 1, ymin)], fill=color)
        draw.text((xmin + 1, ymin - th), text, fill=(255, 255, 255))

    return image


def save_result(save_path, results, catid2name, threshold):
    """
    save result as txt
    """
    img_id = int(results["im_id"])
    with open(save_path, 'w') as f:
        if "bbox_res" in results:
            for dt in results["bbox_res"]:
                catid, bbox, score = dt['category_id'], dt['bbox'], dt['score']
                if score < threshold:
                    continue
                # each bbox result as a line
                # for rbox: classname score x1 y1 x2 y2 x3 y3 x4 y4
                # for bbox: classname score x1 y1 w h
                bbox_pred = '{} {} '.format(catid2name[catid],
                                            score) + ' '.join(
                                                [str(e) for e in bbox])
                f.write(bbox_pred + '\n')
        elif "keypoint_res" in results:
            for dt in results["keypoint_res"]:
                kpts = dt['keypoints']
                scores = dt['score']
                keypoint_pred = [img_id, scores, kpts]
                print(keypoint_pred, file=f)
        else:
            print("No valid results found, skip txt save")
