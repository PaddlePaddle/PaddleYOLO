#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import ppdet


class TestYOLOv3(unittest.TestCase):
    def setUp(self):
        self.set_config()

    def set_config(self):
        self.cfg_file = 'configs/yolov3/yolov3_darknet53_270e_coco.yml'

    def test_trainer(self):
        # Trainer __init__ will build model and DataLoader
        # 'train' and 'eval' mode include dataset loading
        # use 'test' mode to simplify tests
        cfg = ppdet.core.workspace.load_config(self.cfg_file)
        trainer = ppdet.engine.Trainer(cfg, mode='test')


class TestPPYOLOE(TestYOLOv3):
    def set_config(self):
        self.cfg_file = 'configs/ppyoloe/ppyoloe_plus_crn_s_80e_coco.yml'


class TestYOLOX(TestYOLOv3):
    def set_config(self):
        self.cfg_file = 'configs/yolox/yolox_s_300e_coco.yml'


class TestYOLOv5(TestYOLOv3):
    def set_config(self):
        self.cfg_file = 'configs/yolov5/yolov5_s_300e_coco.yml'


class TestYOLOv6(TestYOLOv3):
    def set_config(self):
        self.cfg_file = 'configs/yolov6/yolov6_s_400e_coco.yml'


class TestYOLOv7(TestYOLOv3):
    def set_config(self):
        self.cfg_file = 'configs/yolov7/yolov7_l_300e_coco.yml'


if __name__ == '__main__':
    unittest.main()
