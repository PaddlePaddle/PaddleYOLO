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

from ppdet.data.source.voc import pascalvoc_label
from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

__all__ = ['get_categories']


def get_categories(metric_type, anno_file=None, arch=None):
    """
    Get class id to category id map and category id
    to category name map from annotation file.

    Args:
        metric_type (str): metric type, currently support 'coco', 'voc', 'oid'
            and 'widerface'.
        anno_file (str): annotation file path
    """
    if anno_file == None or (not os.path.isfile(anno_file)):
        logger.warning(
            "anno_file '{}' is None or not set or not exist, "
            "please recheck TrainDataset/EvalDataset/TestDataset.anno_path, "
            "otherwise the default categories will be used by metric_type.".
            format(anno_file))

    if metric_type.lower() == 'coco':
        if anno_file and os.path.isfile(anno_file):
            if anno_file.endswith('json'):
                # lazy import pycocotools here
                from pycocotools.coco import COCO
                coco = COCO(anno_file)
                cats = coco.loadCats(coco.getCatIds())

                clsid2catid = {i: cat['id'] for i, cat in enumerate(cats)}
                catid2name = {cat['id']: cat['name'] for cat in cats}

            elif anno_file.endswith('txt'):
                cats = []
                with open(anno_file) as f:
                    for line in f.readlines():
                        cats.append(line.strip())
                if cats[0] == 'background': cats = cats[1:]

                clsid2catid = {i: i for i in range(len(cats))}
                catid2name = {i: name for i, name in enumerate(cats)}

            else:
                raise ValueError("anno_file {} should be json or txt.".format(
                    anno_file))
            return clsid2catid, catid2name

        # anno file not exist, load default categories of COCO17
        else:
            logger.warning("metric_type: {}, load default categories of COCO.".
                           format(metric_type))
            return _coco17_category()

    elif metric_type.lower() == 'voc':
        if anno_file and os.path.isfile(anno_file):
            cats = []
            with open(anno_file) as f:
                for line in f.readlines():
                    cats.append(line.strip())

            if cats[0] == 'background':
                cats = cats[1:]

            clsid2catid = {i: i for i in range(len(cats))}
            catid2name = {i: name for i, name in enumerate(cats)}

            return clsid2catid, catid2name

        # anno file not exist, load default categories of
        # VOC all 20 categories
        else:
            logger.warning("metric_type: {}, load default categories of VOC.".
                           format(metric_type))
            return _vocall_category()

    else:
        raise ValueError("unknown metric type {}".format(metric_type))


def _coco17_category():
    """
    Get class id to category id map and category id
    to category name map of COCO2017 dataset

    """
    clsid2catid = {
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 8,
        9: 9,
        10: 10,
        11: 11,
        12: 13,
        13: 14,
        14: 15,
        15: 16,
        16: 17,
        17: 18,
        18: 19,
        19: 20,
        20: 21,
        21: 22,
        22: 23,
        23: 24,
        24: 25,
        25: 27,
        26: 28,
        27: 31,
        28: 32,
        29: 33,
        30: 34,
        31: 35,
        32: 36,
        33: 37,
        34: 38,
        35: 39,
        36: 40,
        37: 41,
        38: 42,
        39: 43,
        40: 44,
        41: 46,
        42: 47,
        43: 48,
        44: 49,
        45: 50,
        46: 51,
        47: 52,
        48: 53,
        49: 54,
        50: 55,
        51: 56,
        52: 57,
        53: 58,
        54: 59,
        55: 60,
        56: 61,
        57: 62,
        58: 63,
        59: 64,
        60: 65,
        61: 67,
        62: 70,
        63: 72,
        64: 73,
        65: 74,
        66: 75,
        67: 76,
        68: 77,
        69: 78,
        70: 79,
        71: 80,
        72: 81,
        73: 82,
        74: 84,
        75: 85,
        76: 86,
        77: 87,
        78: 88,
        79: 89,
        80: 90
    }

    catid2name = {
        0: 'background',
        1: 'person',
        2: 'bicycle',
        3: 'car',
        4: 'motorcycle',
        5: 'airplane',
        6: 'bus',
        7: 'train',
        8: 'truck',
        9: 'boat',
        10: 'traffic light',
        11: 'fire hydrant',
        13: 'stop sign',
        14: 'parking meter',
        15: 'bench',
        16: 'bird',
        17: 'cat',
        18: 'dog',
        19: 'horse',
        20: 'sheep',
        21: 'cow',
        22: 'elephant',
        23: 'bear',
        24: 'zebra',
        25: 'giraffe',
        27: 'backpack',
        28: 'umbrella',
        31: 'handbag',
        32: 'tie',
        33: 'suitcase',
        34: 'frisbee',
        35: 'skis',
        36: 'snowboard',
        37: 'sports ball',
        38: 'kite',
        39: 'baseball bat',
        40: 'baseball glove',
        41: 'skateboard',
        42: 'surfboard',
        43: 'tennis racket',
        44: 'bottle',
        46: 'wine glass',
        47: 'cup',
        48: 'fork',
        49: 'knife',
        50: 'spoon',
        51: 'bowl',
        52: 'banana',
        53: 'apple',
        54: 'sandwich',
        55: 'orange',
        56: 'broccoli',
        57: 'carrot',
        58: 'hot dog',
        59: 'pizza',
        60: 'donut',
        61: 'cake',
        62: 'chair',
        63: 'couch',
        64: 'potted plant',
        65: 'bed',
        67: 'dining table',
        70: 'toilet',
        72: 'tv',
        73: 'laptop',
        74: 'mouse',
        75: 'remote',
        76: 'keyboard',
        77: 'cell phone',
        78: 'microwave',
        79: 'oven',
        80: 'toaster',
        81: 'sink',
        82: 'refrigerator',
        84: 'book',
        85: 'clock',
        86: 'vase',
        87: 'scissors',
        88: 'teddy bear',
        89: 'hair drier',
        90: 'toothbrush'
    }

    clsid2catid = {k - 1: v for k, v in clsid2catid.items()}
    catid2name.pop(0)

    return clsid2catid, catid2name


def _vocall_category():
    """
    Get class id to category id map and category id
    to category name map of mixup voc dataset

    """
    label_map = pascalvoc_label()
    label_map = sorted(label_map.items(), key=lambda x: x[1])
    cats = [l[0] for l in label_map]

    clsid2catid = {i: i for i in range(len(cats))}
    catid2name = {i: name for i, name in enumerate(cats)}

    return clsid2catid, catid2name
