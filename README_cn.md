## 简介

此代码库是基于[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)的YOLO系列模型库，支持`PP-YOLOE`,`YOLOv3`,`YOLOX`,`YOLOv5`,`MT-YOLOv6`,`YOLOv7`等模型，其upstream为PaddleDetection的[develop](https://github.com/PaddlePaddle/PaddleDetection/tree/develop)分支，并与PaddleDetection主代码库分支同步更新，包括github和gitee的代码，欢迎一起使用和建设！

**注意:**
 - github链接为：https://github.com/nemonameless/PaddleDetection_YOLOSeries
 - gitee链接为：https://gitee.com/nemonameless/PaddleDetection_YOLOSeries
 - [PP-YOLOE](configs/ppyoloe),[PP-YOLO](configs/ppyolo),[PP-YOLOv2](configs/ppyolo),[YOLOv3](configs/yolov3)和[YOLOX](configs/yolox)等模型推荐在[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)中使用。
 - [YOLOv5](configs/yolox),[YOLOv7](configs/yolov7)和[MT-YOLOv6](configs/yolov6mt)模型推荐在此代码库中使用，由于GPL开源协议不合入PaddleDetection主代码库。


## ModelZoo on COCO

### [PP-YOLOE](configs/ppyoloe)

- model zoo

| 网络网络        | 输入尺寸   | 图片数/GPU | 学习率策略 | 推理耗时(ms) | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | Params(M) | FLOPs(G) |    下载链接       | 配置文件 |
| :------------- | :------- | :-------: | :------: | :------------: | :---------------------: | :----------------: |:---------: | :------: |:---------------: |:-----: |
| PP-YOLOE-s   |     640   |    32    |  400e    |    2.9    |       43.4        |        60.0         |   7.93    |  17.36   | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_s_400e_coco.pdparams) | [config](configs/ppyoloe/ppyoloe_crn_s_400e_coco.yml)                   |
| PP-YOLOE-s   |     640   |    32    |  300e    |    2.9    |       43.0        |        59.6         |   7.93    |  17.36   | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_s_300e_coco.pdparams) | [config](configs/ppyoloe/ppyoloe_crn_s_300e_coco.yml)                   |
| PP-YOLOE-m   |      640  |    28    |  300e    |    6.0    |       49.0        |        49.1         |   23.43   |  49.91   | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_m_300e_coco.pdparams) | [config](configs/ppyoloe/ppyoloe_crn_m_300e_coco.yml)                   |
| PP-YOLOE-l   |      640  |    20    |  300e    |    8.7    |       51.4        |        65.9         |   52.20   |  110.07 | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams) | [config](configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml)                   |
| PP-YOLOE-x   |      640  |    16    |  300e    |    14.9   |       **52.3**    |        **69.5**     |   98.42   |  206.59  |[model](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_x_300e_coco.pdparams) | [config](configs/ppyoloe/ppyoloe_crn_x_300e_coco.yml)  
| PP-YOLOE-ConvNeXt-tiny | 640 |    16      |   36e    | -   |       44.6        |        63.3         |   33.04   |  **13.87** | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_convnext_tiny_36e_coco.pdparams) | [config](configs/convnext/ppyoloe_convnext_tiny_36e_coco.yml) |

 
- 部署模型
 
| 模型     | 输入尺寸 | 导出后的模型(w/o 后处理) | ONNX(w/o 后处理) |
| :-------- | :--------: | :---------------------: | :----------------: |
| PP-YOLOE-s(400epoch) |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/ppyoloe_w_nms/ppyoloe_crn_s_400e_coco.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/ppyoloe_wo_nms/ppyoloe_crn_s_400e_coco.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/ppyoloe_crn_s_400e_coco_postprocessed.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/ppyoloe_crn_s_400e_coco.onnx) |
| PP-YOLOE-s |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/ppyoloe_w_nms/ppyoloe_crn_s_300e_coco.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/ppyoloe_wo_nms/ppyoloe_crn_s_300e_coco.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/ppyoloe_crn_s_300e_coco_postprocessed.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/ppyoloe_crn_s_300e_coco.onnx) |
| PP-YOLOE-m |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/ppyoloe_w_nms/ppyoloe_crn_m_300e_coco.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/ppyoloe_wo_nms/ppyoloe_crn_m_300e_coco.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/ppyoloe_crn_m_300e_coco_postprocessed.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/ppyoloe_crn_m_300e_coco.onnx) |
| PP-YOLOE-l |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/ppyoloe_w_nms/ppyoloe_crn_l_300e_coco.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/ppyoloe_wo_nms/ppyoloe_crn_l_300e_coco.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/ppyoloe_crn_l_300e_coco_postprocessed.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/ppyoloe_crn_l_300e_coco.onnx) |
| PP-YOLOE-x |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/ppyoloe_w_nms/ppyoloe_crn_x_300e_coco.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/ppyoloe_wo_nms/ppyoloe_crn_x_300e_coco.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/ppyoloe_crn_x_300e_coco_postprocessed.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/ppyoloe_crn_x_300e_coco.onnx) |
 
### [YOLOX](configs/yolox)

| 网络网络        | 输入尺寸   | 图片数/GPU | 学习率策略 | 推理耗时(ms) | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | Params(M) | FLOPs(G) |    下载链接       | 配置文件 |
| :------------- | :------- | :-------: | :------: | :------------: | :---------------------: | :----------------: |:---------: | :------: |:---------------: |:-----: |
| YOLOX-nano     |  416     |    8      |   300e    |     2.3    |  26.1  |  42.0 |  0.91  |  1.08 | [model](https://paddledet.bj.bcebos.com/models/yolox_nano_300e_coco.pdparams) | [config](configs/yolox/yolox_nano_300e_coco.yml) |
| YOLOX-tiny     |  416     |    8      |   300e    |     2.8    |  32.9  |  50.4 |  5.06  |  6.45 | [model](https://paddledet.bj.bcebos.com/models/yolox_tiny_300e_coco.pdparams) | [config](configs/yolox/yolox_tiny_300e_coco.yml) |
| YOLOX-s        |  640     |    8      |   300e    |     3.0    |  40.4  |  59.6 |  9.0  |  26.8 | [model](https://paddledet.bj.bcebos.com/models/yolox_s_300e_coco.pdparams) | [config](configs/yolox/yolox_s_300e_coco.yml) |
| YOLOX-m        |  640     |    8      |   300e    |     5.8    |  46.9  |  65.7 |  25.3  |  73.8 | [model](https://paddledet.bj.bcebos.com/models/yolox_m_300e_coco.pdparams) | [config](configs/yolox/yolox_m_300e_coco.yml) |
| YOLOX-l        |  640     |    8      |   300e    |     9.3    |  50.1  |  68.8 |  54.2  |  155.6 | [model](https://paddledet.bj.bcebos.com/models/yolox_l_300e_coco.pdparams) | [config](configs/yolox/yolox_l_300e_coco.yml) |
| YOLOX-x        |  640     |    8      |   300e    |     16.6   |  **51.8**  |  **70.6** |  99.1  |  281.9 | [model](https://paddledet.bj.bcebos.com/models/yolox_x_300e_coco.pdparams) | [config](configs/yolox/yolox_x_300e_coco.yml) |
 YOLOX-cdn-tiny    |  416     |    8      |   300e    |     1.9    |  32.4  |  50.2 |  5.03 |  6.33  | [model](https://paddledet.bj.bcebos.com/models/yolox_cdn_tiny_300e_coco.pdparams) | [config](configs/yolox/yolox_cdn_tiny_300e_coco.yml) |
| YOLOX-crn-s     |  640     |    8      |   300e    |     3.0    |  40.4  |  59.6 |  7.7  |  24.69 | [model](https://paddledet.bj.bcebos.com/models/yolox_crn_s_300e_coco.pdparams) | [config](configs/yolox/yolox_crn_s_300e_coco.yml) |
| YOLOX-ConvNeXt-s|  640     |    8      |   36e     |     -      |  44.6  |  65.3 |  36.2 |  27.52 | [model](https://paddledet.bj.bcebos.com/models/yolox_convnext_s_36e_coco.pdparams) | [config](configs/convnext/yolox_convnext_s_36e_coco.yml) |

### [YOLOv5](configs/yolov5)

| 网络网络        | 输入尺寸   | 图片数/GPU | 学习率策略 | 推理耗时(ms) | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | Params(M) | FLOPs(G) |    下载链接       | 配置文件 |
| :------------- | :------- | :-------: | :------: | :------------: | :---------------------: | :----------------: |:---------: | :------: |:---------------: |:-----: |
| YOLOv5-n        |  640     |    16     |   300e    |     2.6    |  28.0  | 45.7 |  1.87  | 4.52 | [model](https://paddledet.bj.bcebos.com/models/yolov5_n_300e_coco.pdparams) | [config](configs/yolov5/yolov5_n_300e_coco.yml) |
| YOLOv5-s        |  640     |    8      |   300e    |     3.2    |  37.0  | 55.9 |  7.24  | 16.54 | [model](https://paddledet.bj.bcebos.com/models/yolov5_s_300e_coco.pdparams) | [config](configs/yolov5/yolov5_s_300e_coco.yml) |
| YOLOv5-m        |  640     |    8      |   300e    |     5.2    |  45.3  | 63.8 |  21.19  | 49.08 | [model](https://paddledet.bj.bcebos.com/models/yolov5_m_300e_coco.pdparams) | [config](configs/yolov5/yolov5_m_300e_coco.yml) |
| YOLOv5-l        |  640     |    8      |   300e    |     7.9    |  48.6  | 66.9 |  46.56  | 109.32 | [model](https://paddledet.bj.bcebos.com/models/yolov5_l_300e_coco.pdparams) | [config](configs/yolov5/yolov5_l_300e_coco.yml) |
| YOLOv5-x        |  640     |    8      |   300e    |     13.7    |  **50.6**  | **68.7** |  86.75  | 205.92 | [model](https://paddledet.bj.bcebos.com/models/yolov5_x_300e_coco.pdparams) | [config](configs/yolov5/yolov5_x_300e_coco.yml) |
| YOLOv5-ConvNeXt-s|  640    |    8      |   36e     |     -      |  42.4  |  65.3  |  34.54 |  17.96 | [model](https://paddledet.bj.bcebos.com/models/yolov5_convnext_s_36e_coco.pdparams) | [config](configs/yolov5/yolov5_convnext_s_36e_coco.yml) |

### [MT-YOLOv6](configs/yolov6mt)

| 网络网络        | 输入尺寸   | 图片数/GPU | 学习率策略 | 推理耗时(ms) |   mAP  |   AP50  | Params(M) | FLOPs(G) |  下载链接       | 配置文件 |
| :------------- | :------- | :-------: | :------: | :---------: | :-----: |:-----: | :-----: |:-----: | :-------------: | :-----: |
| *YOLOv6mt-n       |  416     |    32      |   400e    |     2.5    | 30.5  |    46.8 |  4.74  | 5.16 |[model](https://paddledet.bj.bcebos.com/models/yolov6mt_n_416_400e_coco.pdparams) | [config](configs/yolov6mt/yolov6mt_n_416_400e_coco.yml) |
| *YOLOv6mt-n       |  640     |    32      |   400e    |     2.8    |  34.7 |    52.7 |  4.74  |  12.2 |[model](https://paddledet.bj.bcebos.com/models/yolov6mt_n_400e_coco.pdparams) | [config](configs/yolov6mt/yolov6mt_n_400e_coco.yml) |
| *YOLOv6mt-t       |  640     |    32      |   400e    |     2.9    |  40.8 |  60.4 |  16.36  | 39.94 |[model](https://paddledet.bj.bcebos.com/models/yolov6mt_t_400e_coco.pdparams) | [config](configs/yolov6mt/yolov6mt_t_400e_coco.yml) |
| *YOLOv6mt-s       |  640     |    32      |   400e    |     3.0    | 42.5 |    61.7 |  18.87  | 48.36 |[model](https://paddledet.bj.bcebos.com/models/yolov6mt_s_400e_coco.pdparams) | [config](configs/yolov6mt/yolov6mt_s_400e_coco.yml) |

### [YOLOv7](configs/yolov7)

| 网络网络        | 输入尺寸   | 图片数/GPU | 学习率策略 | 推理耗时(ms) | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | Params(M) | FLOPs(G) |    下载链接       | 配置文件 |
| :------------- | :------- | :-------: | :------: | :------------: | :---------------------: | :----------------: |:---------: | :------: |:---------------: |:-----: |
| YOLOv7-L        |  640     |    32      |   300e    |     7.4     |  51.0  | 70.2 |  37.62  | 106.08 |[model](https://paddledet.bj.bcebos.com/models/yolov7_l_300e_coco.pdparams) | [config](configs/yolov7/yolov7_l_300e_coco.yml) |
| *YOLOv7-X        |  640     |    32      |   300e    |     12.2    |  53.0  | 70.8 |  71.34  | 190.08 | [model](https://paddledet.bj.bcebos.com/models/yolov7_x_300e_coco.pdparams) | [config](configs/yolov7/yolov7_x_300e_coco.yml) |
| *YOLOv7P6-W6     |  1280    |    16      |   300e    |     25.5    |  54.4  | 71.8 |  70.43  | 360.26 | [model](https://paddledet.bj.bcebos.com/models/yolov7p6_w6_300e_coco.pdparams) | [config](configs/yolov7/yolov7p6_w6_300e_coco.yml) |
| *YOLOv7P6-E6     |  1280    |    10      |   300e    |     31.1    |  55.7  | 73.0 |  97.25  | 515.4 | [model](https://paddledet.bj.bcebos.com/models/yolov7p6_e6_300e_coco.pdparams) | [config](configs/yolov7/yolov7p6_e6_300e_coco.yml) |
| *YOLOv7P6-D6     |  1280    |    8      |   300e    |     37.4    | 56.1  | 73.3 |  133.81  | 702.92 | [model](https://paddledet.bj.bcebos.com/models/yolov7p6_d6_300e_coco.pdparams) | [config](configs/yolov7/yolov7p6_d6_300e_coco.yml) |
| *YOLOv7P6-E6E    |  1280    |    6      |   300e    |     48.7    |  56.5  | 73.7 |  151.76  | 843.52 | [model](https://paddledet.bj.bcebos.com/models/yolov7p6_e6e_300e_coco.pdparams) | [config](configs/yolov7/yolov7p6_e6e_300e_coco.yml) |


**注意:**
 - 所有模型均使用COCO train2017作为训练集，在COCO val2017上验证精度，模型前带*表示训练更新中。
 - 具体精度和速度细节请查看[PP-YOLOE](configs/ppyoloe),[YOLOX](configs/yolox),[YOLOv5](configs/yolox),[MT-YOLOv6](configs/yolov6mt),[YOLOv7](configs/yolov7)。
- 模型推理耗时(ms)为TensorRT-FP16下测试的耗时，不包含数据预处理和模型输出后处理(NMS)的耗时。测试采用单卡V100，batch size=1，测试环境为**paddlepaddle-2.3.0**, **CUDA 11.2**, **CUDNN 8.2**, **GCC-8.2**, **TensorRT 8.0.3.4**，具体请参考各自模型主页。
 - [PP-YOLOE](configs/ppyoloe),[YOLOv3](configs/yolov3)和[YOLOX](configs/yolox)推荐在[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)里使用。
 - [YOLOv5](configs/yolox),[YOLOv7](configs/yolov7)和[MT-YOLOv6](configs/yolov6mt)由于GPL协议而不合入[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)主代码库。
 - 各模型导出后的权重以及ONNX，分为带不带后处理NMS，都提供了下载链接，请参考各自模型主页下载。
 - paddlepaddle版本推荐使用2.3.0以上。


## 使用指南

### COCO数据集下载
下载MS COCO数据集，[官网](https://cocodataset.org)下载地址为: [annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip), [train2017](http://images.cocodataset.org/zips/train2017.zip), [val2017](http://images.cocodataset.org/zips/val2017.zip), [test2017](http://images.cocodataset.org/zips/test2017.zip)。
PaddleDetection团队提供的下载链接为：[coco](https://bj.bcebos.com/v1/paddledet/data/coco.tar)(共约22G)和[test2017](https://bj.bcebos.com/v1/paddledet/data/cocotest2017.zip)，注意test2017可不下载，评估是使用的val2017。


### **一键运行全流程**:
```
model_type=ppyoloe # 可修改，如 yolov7
job_name=ppyoloe_crn_l_300e_coco # 可修改，如 yolov7_l_300e_coco

config=configs/${model_type}/${job_name}.yml
log_dir=log_dir/${job_name}
# weights=https://bj.bcebos.com/v1/paddledet/models/${job_name}.pdparams
weights=output/${job_name}/model_final.pdparams

# 1.训练（单卡/多卡）
# CUDA_VISIBLE_DEVICES=0 python3.7 tools/train.py -c ${config} --eval --amp
python3.7 -m paddle.distributed.launch --log_dir=${log_dir} --gpus 0,1,2,3,4,5,6,7 tools/train.py -c ${config} --eval --amp

# 2.评估
CUDA_VISIBLE_DEVICES=0 python3.7 tools/eval.py -c ${config} -o weights=${weights} --classwise

# 3.直接预测
CUDA_VISIBLE_DEVICES=0 python3.7 tools/infer.py -c ${config} -o weights=${weights} --infer_img=demo/000000014439_640x640.jpg --draw_threshold=0.5

# 4.导出模型
CUDA_VISIBLE_DEVICES=0 python3.7 tools/export_model.py -c ${config} -o weights=${weights} # exclude_nms=True trt=True

# 5.部署预测
CUDA_VISIBLE_DEVICES=0 python3.7 deploy/python/infer.py --model_dir=output_inference/${job_name} --image_file=demo/000000014439_640x640.jpg --device=GPU

# 6.部署测速
CUDA_VISIBLE_DEVICES=0 python3.7 deploy/python/infer.py --model_dir=output_inference/${job_name} --image_file=demo/000000014439_640x640.jpg --device=GPU --run_benchmark=True # --run_mode=trt_fp16

# 7.onnx导出
paddle2onnx --model_dir output_inference/${job_name} --model_filename model.pdmodel --params_filename model.pdiparams --opset_version 12 --save_file ${job_name}.onnx

# 8.onnx测速
/usr/local/TensorRT-8.0.3.4/bin/trtexec --onnx=${job_name}.onnx --workspace=4096 --avgRuns=10 --shapes=input:1x3x640x640 --fp16

```

**注意:**
- 将以上命令写在一个脚本文件里如```run.sh```，一键运行命令为：```sh run.sh```，也可命令行一句句去运行。
- 如果想切换模型，只要修改开头两行即可，如:
  ```
  model_type=yolov7
  job_name=yolov7_l_300e_coco
  ```

### 自定义数据集训练：

#### 数据集准备：

1.自定义数据集的标注制作，请参考[DetAnnoTools](docs/tutorials/data/DetAnnoTools.md);

2.自定义数据集的训练准备，请参考[PrepareDataSet](docs/tutorials/PrepareDataSet.md)。


#### fintune训练：

除了更改数据集的路径外，训练一般推荐加载**对应模型的COCO预训练权重**去fintune，会更快收敛和达到更高精度，如：

```base
# 单卡fintune训练：
# CUDA_VISIBLE_DEVICES=0 python3.7 tools/train.py -c ${config} --eval --amp -o pretrain_weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams

# 多卡fintune训练：
python3.7 -m paddle.distributed.launch --log_dir=./log_dir --gpus 0,1,2,3,4,5,6,7 tools/train.py -c ${config} --eval --amp -o pretrain_weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams
```

**注意:**
- fintune训练一般会提示head分类分支最后一层卷积的通道数没对应上，属于正常情况，是由于自定义数据集一般和COCO数据集种类数不一致；
- fintune训练一般epoch数可以设置更少，lr设置也更小点如1/10，最高精度可能出现在中间某个epoch；

#### 预测和导出：

使用自定义数据集预测和导出模型时，如果TestDataset数据集路径设置不正确会默认使用COCO 80类。
除了TestDataset数据集路径设置正确外，也可以自行修改和添加对应的label_list.txt文件(一行记录一个对应种类)，TestDataset中的anno_path也可设置为绝对路径，如：
```
TestDataset:
  !ImageFolder
    anno_path: label_list.txt # 如不使用dataset_dir，则anno_path即为相对于PaddleDetection主目录的相对路径
    # dataset_dir: dataset/my_coco # 如使用dataset_dir，则dataset_dir/anno_path作为新的anno_path
```
label_list.txt里的一行记录一个对应种类，如下所示：
```
person
vehicle
```

======================================================


简体中文 | [English](README_en.md)

<div align="center">
<p align="center">
  <img src="https://user-images.githubusercontent.com/48054808/160532560-34cf7a1f-d950-435e-90d2-4b0a679e5119.png" align="middle" width = "800" />
</p>

**飞桨目标检测开发套件，端到端地完成从训练到部署的全流程目标检测应用。**

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleDetection/releases"><img src="https://img.shields.io/github/v/release/PaddlePaddle/PaddleDetection?color=ffa"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleDetection/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/PaddleDetection?color=ccf"></a>
</p>
</div>

<div  align="center">
  <img src="docs/images/ppdet.gif" width="800"/>

</div>

## <img src="https://user-images.githubusercontent.com/48054808/157793354-6e7f381a-0aa6-4bb7-845c-9acf2ecc05c3.png" width="20"/> 产品动态

- 🔥 **2022.7.14：[行人分析工具PP-Human v2](./deploy/pipeline)发布**
  - 四大产业特色功能：高性能易扩展的五大复杂行为识别、闪电级人体属性识别、一行代码即可实现的人流检测与轨迹留存以及高精度跨镜跟踪
  - 底层核心算法性能强劲：覆盖行人检测、跟踪、属性三类核心算法能力，对目标人数、光线、背景均无限制
  - 极低使用门槛：提供保姆级全流程开发及模型优化策略、一行命令完成推理、兼容各类数据输入格式

**活动预告** 7月19日晚20点，PaddleDetection举办PP-Human v2线上私享交流会，欢迎大家扫码进群，获取线上会议链接！名额有限，抓紧报名！
<div  align="center">
  <img src="https://user-images.githubusercontent.com/22989727/178771163-66639dc0-cb65-4413-8de4-6ac5c5eed9f5.jpg" width="200"/>
</div>

- 2022.3.24：PaddleDetection发布[release/2.4版本](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4)
  - 发布高精度云边一体SOTA目标检测模型[PP-YOLOE](configs/ppyoloe)，提供s/m/l/x版本，l版本COCO test2017数据集精度51.6%，V100预测速度78.1 FPS，支持混合精度训练，训练较PP-YOLOv2加速33%，全系列多尺度模型，满足不同硬件算力需求，可适配服务器、边缘端GPU及其他服务器端AI加速卡。
  - 发布边缘端和CPU端超轻量SOTA目标检测模型[PP-PicoDet增强版](configs/picodet)，精度提升2%左右，CPU预测速度提升63%，新增参数量0.7M的PicoDet-XS模型，提供模型稀疏化和量化功能，便于模型加速，各类硬件无需单独开发后处理模块，降低部署门槛。
  - 发布实时行人分析工具[PP-Human](deploy/pipeline)，支持行人跟踪、人流量统计、人体属性识别与摔倒检测四大能力，基于真实场景数据特殊优化，精准识别各类摔倒姿势，适应不同环境背景、光线及摄像角度。
  - 新增[YOLOX](configs/yolox)目标检测模型，支持nano/tiny/s/m/l/x版本，x版本COCO val2017数据集精度51.8%。

- 2021.11.03: PaddleDetection发布[release/2.3版本](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3)

  - 发布轻量级检测特色模型⚡[PP-PicoDet](configs/picodet)，0.99m的参数量可实现精度30+mAP、速度150FPS。
  - 发布轻量级关键点特色模型⚡[PP-TinyPose](configs/keypoint/tiny_pose)，单人场景FP16推理可达122FPS、51.8AP，具有精度高速度快、检测人数无限制、微小目标效果好的优势。
  - 发布实时跟踪系统[PP-Tracking](deploy/pptracking)，覆盖单、多镜头下行人、车辆、多类别跟踪，对小目标、密集型特殊优化，提供人、车流量技术解决方案。
  - 新增[Swin Transformer](configs/faster_rcnn)，[TOOD](configs/tood)，[GFL](configs/gfl)目标检测模型。
  - 发布[Sniper](configs/sniper)小目标检测优化模型，发布针对EdgeBoard优化[PP-YOLO-EB](configs/ppyolo)模型。
  - 新增轻量化关键点模型[Lite HRNet](configs/keypoint)关键点模型并支持Paddle Lite部署。

- [更多版本发布](https://github.com/PaddlePaddle/PaddleDetection/releases)

## <img title="" src="https://user-images.githubusercontent.com/48054808/157795569-9fc77c85-732f-4870-9be0-99a7fe2cff27.png" alt="" width="20"> 简介

**PaddleDetection**为基于飞桨PaddlePaddle的端到端目标检测套件，内置**30+模型算法**及**250+预训练模型**，覆盖**目标检测、实例分割、跟踪、关键点检测**等方向，其中包括**服务器端和移动端高精度、轻量级**产业级SOTA模型、冠军方案和学术前沿算法，并提供配置化的网络模块组件、十余种数据增强策略和损失函数等高阶优化支持和多种部署方案，在打通数据处理、模型开发、训练、压缩、部署全流程的基础上，提供丰富的案例及教程，加速算法产业落地应用。

<div  align="center">
  <img src="https://user-images.githubusercontent.com/48054808/157826886-2e101a71-25a2-42f5-bf5e-30a97be28f46.gif" width="800"/>
</div>

## <img src="https://user-images.githubusercontent.com/48054808/157799599-e6a66855-bac6-4e75-b9c0-96e13cb9612f.png" width="20"/> 特性

- **模型丰富**: 包含**目标检测**、**实例分割**、**人脸检测**、****关键点检测****、**多目标跟踪**等**250+个预训练模型**，涵盖多种**全球竞赛冠军**方案。
- **使用简洁**：模块化设计，解耦各个网络组件，开发者轻松搭建、试用各种检测模型及优化策略，快速得到高性能、定制化的算法。
- **端到端打通**: 从数据增强、组网、训练、压缩、部署端到端打通，并完备支持**云端**/**边缘端**多架构、多设备部署。
- **高性能**: 基于飞桨的高性能内核，模型训练速度及显存占用优势明显。支持FP16训练, 支持多机训练。

<div  align="center">
  <img src="https://user-images.githubusercontent.com/48054808/172783897-26a93368-d262-443c-a838-8f36bfd714e5.png" width="800"/>
</div>

## <img title="" src="https://user-images.githubusercontent.com/48054808/157800467-2a9946ad-30d1-49a9-b9db-ba33413d9c90.png" alt="" width="20"> 技术交流

- 如果你发现任何PaddleDetection存在的问题或者是建议, 欢迎通过[GitHub Issues](https://github.com/PaddlePaddle/PaddleDetection/issues)给我们提issues。

- 欢迎加入PaddleDetection QQ、微信用户群（添加并回复小助手“检测”）

  <div align="center">
  <img src="https://user-images.githubusercontent.com/48054808/157800129-2f9a0b72-6bb8-4b10-8310-93ab1639253f.jpg"  width = "200" />  
  <img src="https://user-images.githubusercontent.com/34162360/177678712-4655747d-4290-4ad9-b7a1-4564a5418ac6.jpg"  width = "200" />  
  </div>

## <img src="https://user-images.githubusercontent.com/48054808/157827140-03ffaff7-7d14-48b4-9440-c38986ea378c.png" width="20"/> 套件结构概览

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Architectures</b>
      </td>
      <td>
        <b>Backbones</b>
      </td>
      <td>
        <b>Components</b>
      </td>
      <td>
        <b>Data Augmentation</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
        <details><summary><b>Object Detection</b></summary>
          <ul>
            <li>Faster RCNN</li>
            <li>FPN</li>
            <li>Cascade-RCNN</li>
            <li>PSS-Det</li>
            <li>RetinaNet</li>
            <li>YOLOv3</li>  
            <li>PP-YOLOv1/v2</li>
            <li>PP-YOLO-Tiny</li>
            <li>PP-YOLOE</li>
            <li>YOLOX</li>
            <li>SSD</li>
            <li>CenterNet</li>
            <li>FCOS</li>  
            <li>TTFNet</li>
            <li>TOOD</li>
            <li>GFL</li>
            <li>PP-PicoDet</li>
            <li>DETR</li>
            <li>Deformable DETR</li>
            <li>Swin Transformer</li>
            <li>Sparse RCNN</li>
         </ul></details>
        <details><summary><b>Instance Segmentation</b></summary>
         <ul>
            <li>Mask RCNN</li>
            <li>Cascade Mask RCNN</li>
            <li>SOLOv2</li>
        </ul></details>
        <details><summary><b>Face Detection</b></summary>
        <ul>
            <li>BlazeFace</li>
        </ul></details>
        <details><summary><b>Multi-Object-Tracking</b></summary>
        <ul>
            <li>JDE</li>
            <li>FairMOT</li>
            <li>DeepSORT</li>
            <li>ByteTrack</li>
        </ul></details>
        <details><summary><b>KeyPoint-Detection</b></summary>
        <ul>
            <li>HRNet</li>
            <li>HigherHRNet</li>
            <li>Lite-HRNet</li>
            <li>PP-TinyPose</li>
        </ul></details>
      </ul>
      </td>
      <td>
        <details><summary><b>Details</b></summary>
        <ul>
          <li>ResNet(&vd)</li>
          <li>Res2Net(&vd)</li>
          <li>CSPResNet</li>
          <li>SENet</li>
          <li>Res2Net</li>
          <li>HRNet</li>
          <li>Lite-HRNet</li>
          <li>DarkNet</li>
          <li>CSPDarkNet</li>
          <li>MobileNetv1/v3</li>  
          <li>ShuffleNet</li>
          <li>GhostNet</li>
          <li>BlazeNet</li>
          <li>DLA</li>
          <li>HardNet</li>
          <li>LCNet</li>  
          <li>ESNet</li>  
          <li>Swin-Transformer</li>
        </ul></details>
      </td>
      <td>
        <details><summary><b>Common</b></summary>
          <ul>
            <li>Sync-BN</li>
            <li>Group Norm</li>
            <li>DCNv2</li>
            <li>EMA</li>
          </ul> </details>
        </ul>
        <details><summary><b>KeyPoint</b></summary>
          <ul>
            <li>DarkPose</li>
          </ul></details>
        </ul>
        <details><summary><b>FPN</b></summary>
          <ul>
            <li>BiFPN</li>
            <li>CSP-PAN</li>
            <li>Custom-PAN</li>
            <li>ES-PAN</li>
            <li>HRFPN</li>
          </ul> </details>
        </ul>  
        <details><summary><b>Loss</b></summary>
          <ul>
            <li>Smooth-L1</li>
            <li>GIoU/DIoU/CIoU</li>  
            <li>IoUAware</li>
            <li>Focal Loss</li>
            <li>CT Focal Loss</li>
            <li>VariFocal Loss</li>
          </ul> </details>
        </ul>  
        <details><summary><b>Post-processing</b></summary>
          <ul>
            <li>SoftNMS</li>
            <li>MatrixNMS</li>  
          </ul> </details>  
        </ul>
        <details><summary><b>Speed</b></summary>
          <ul>
            <li>FP16 training</li>
            <li>Multi-machine training </li>  
          </ul> </details>  
        </ul>  
      </td>
      <td>
        <details><summary><b>Details</b></summary>
        <ul>
          <li>Resize</li>  
          <li>Lighting</li>  
          <li>Flipping</li>  
          <li>Expand</li>
          <li>Crop</li>
          <li>Color Distort</li>  
          <li>Random Erasing</li>  
          <li>Mixup </li>
          <li>AugmentHSV</li>
          <li>Mosaic</li>
          <li>Cutmix </li>
          <li>Grid Mask</li>
          <li>Auto Augment</li>  
          <li>Random Perspective</li>  
        </ul> </details>  
      </td>  
    </tr>

</td>
    </tr>
  </tbody>
</table>

## <img src="https://user-images.githubusercontent.com/48054808/157801371-9a9a8c65-1690-4123-985a-e0559a7f9494.png" width="20"/> 模型性能概览

<details>
<summary><b> 云端模型性能对比</b></summary>

各模型结构和骨干网络的代表模型在COCO数据集上精度mAP和单卡Tesla V100上预测速度(FPS)对比图。

<div align="center">
  <img src="docs/images/fps_map.png" />
</div>

**说明：**

- `CBResNet`为`Cascade-Faster-RCNN-CBResNet200vd-FPN`模型，COCO数据集mAP高达53.3%
- `Cascade-Faster-RCNN`为`Cascade-Faster-RCNN-ResNet50vd-DCN`，PaddleDetection将其优化到COCO数据mAP为47.8%时推理速度为20FPS
- `PP-YOLO`在COCO数据集精度45.9%，Tesla V100预测速度72.9FPS，精度速度均优于[YOLOv4](https://arxiv.org/abs/2004.10934)
- `PP-YOLO v2`是对`PP-YOLO`模型的进一步优化，在COCO数据集精度49.5%，Tesla V100预测速度68.9FPS
- `PP-YOLOE`是对`PP-YOLO v2`模型的进一步优化，在COCO数据集精度51.6%，Tesla V100预测速度78.1FPS
- [`YOLOX`](configs/yolox)和[`YOLOv5`](https://github.com/nemonameless/PaddleDetection_YOLOv5/tree/main/configs/yolov5)均为基于PaddleDetection复现算法
- 图中模型均可在[模型库](#模型库)中获取

</details>

<details>
<summary><b> 移动端模型性能对比</b></summary>

各移动端模型在COCO数据集上精度mAP和高通骁龙865处理器上预测速度(FPS)对比图。

<div align="center">
  <img src="docs/images/mobile_fps_map.png" width=600/>
</div>

**说明：**

- 测试数据均使用高通骁龙865(4\*A77 + 4\*A55)处理器batch size为1, 开启4线程测试，测试使用NCNN预测库，测试脚本见[MobileDetBenchmark](https://github.com/JiweiMaster/MobileDetBenchmark)
- [PP-PicoDet](configs/picodet)及[PP-YOLO-Tiny](configs/ppyolo)为PaddleDetection自研模型，其余模型PaddleDetection暂未提供

</details>

## <img src="https://user-images.githubusercontent.com/48054808/157829890-a535b8a6-631c-4c87-b861-64d4b32b2d6a.png" width="20"/> 模型库

<details>
<summary><b> 1. 通用检测</b></summary>

#### [PP-YOLOE](./configs/ppyoloe)系列 推荐场景：Nvidia V100, T4等云端GPU和Jetson系列等边缘端设备

| 模型名称       | COCO精度（mAP） | V100 TensorRT FP16速度(FPS) | 配置文件                                                  | 模型下载                                                                                 |
|:---------- |:-----------:|:-------------------------:|:-----------------------------------------------------:|:------------------------------------------------------------------------------------:|
| PP-YOLOE-s | 42.7        | 333.3                     | [链接](configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml)     | [下载地址](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams)      |
| PP-YOLOE-m | 48.6        | 208.3                     | [链接](configs/ppyolo/ppyolo_r50vd_dcn_2x_coco.yml)     | [下载地址](https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_2x_coco.pdparams)     |
| PP-YOLOE-l | 50.9        | 149.2                     | [链接](configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml) | [下载地址](https://paddledet.bj.bcebos.com/models/ppyolov2_r50vd_dcn_365e_coco.pdparams) |
| PP-YOLOE-x | 51.9        | 95.2                      | [链接](configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml) | [下载地址](https://paddledet.bj.bcebos.com/models/ppyolov2_r50vd_dcn_365e_coco.pdparams) |

#### [PP-PicoDet](./configs/picodet)系列 推荐场景：ARM CPU(RK3399, 树莓派等) 和NPU(比特大陆，晶晨等)移动端芯片和x86 CPU设备

| 模型名称       | COCO精度（mAP） | 骁龙865 四线程速度(ms) | 配置文件                                                | 模型下载                                                                              |
|:---------- |:-----------:|:---------------:|:---------------------------------------------------:|:---------------------------------------------------------------------------------:|
| PicoDet-XS | 23.5        | 7.81            | [链接](configs/picodet/picodet_xs_320_coco_lcnet.yml) | [下载地址](https://paddledet.bj.bcebos.com/models/picodet_xs_320_coco_lcnet.pdparams) |
| PicoDet-S  | 29.1        | 9.56            | [链接](configs/picodet/picodet_s_320_coco_lcnet.yml)  | [下载地址](https://paddledet.bj.bcebos.com/models/picodet_s_320_coco_lcnet.pdparams)  |
| PicoDet-M  | 34.4        | 17.68           | [链接](configs/picodet/picodet_m_320_coco_lcnet.yml)  | [下载地址](https://paddledet.bj.bcebos.com/models/picodet_m_320_coco_lcnet.pdparams)  |
| PicoDet-L  | 36.1        | 25.21           | [链接](configs/picodet/picodet_l_320_coco_lcnet.yml)  | [下载地址](https://paddledet.bj.bcebos.com/models/picodet_l_320_coco_lcnet.pdparams)  |

#### 前沿检测算法

| 模型名称                                                               | COCO精度（mAP） | V100 TensorRT FP16速度(FPS) | 配置文件                                                                                                         | 模型下载                                                                       |
|:------------------------------------------------------------------ |:-----------:|:-------------------------:|:------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------:|
| [YOLOX-l](configs/yolox)                                           | 50.1        | 107.5                     | [链接](configs/yolox/yolox_l_300e_coco.yml)                                                                    | [下载地址](https://paddledet.bj.bcebos.com/models/yolox_l_300e_coco.pdparams)  |
| [YOLOv5-l](https://github.com/nemonameless/PaddleDetection_YOLOv5) | 48.6        | 136.0                     | [链接](https://github.com/nemonameless/PaddleDetection_YOLOv5/blob/main/configs/yolov5/yolov5_l_300e_coco.yml) | [下载地址](https://paddledet.bj.bcebos.com/models/yolov5_l_300e_coco.pdparams) |

#### 其他通用检测模型 [文档链接](docs/MODEL_ZOO_cn.md)

</details>

<details>
<summary><b> 2. 实例分割</b></summary>

| 模型名称              | 模型简介         | 推荐场景 | COCO精度(mAP)                      | 配置文件                                                                  | 模型下载                                                                                              |
|:----------------- |:------------ |:---- |:--------------------------------:|:---------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------:|
| Mask RCNN         | 两阶段实例分割算法    | 云边端  | box AP: 41.4 <br/> mask AP: 37.5 | [链接](configs/mask_rcnn/mask_rcnn_r50_vd_fpn_2x_coco.yml)              | [下载地址](https://paddledet.bj.bcebos.com/models/mask_rcnn_r50_vd_fpn_2x_coco.pdparams)              |
| Cascade Mask RCNN | 两阶段实例分割算法    | 云边端  | box AP: 45.7 <br/> mask AP: 39.7 | [链接](configs/mask_rcnn/cascade_mask_rcnn_r50_vd_fpn_ssld_2x_coco.yml) | [下载地址](https://paddledet.bj.bcebos.com/models/cascade_mask_rcnn_r50_vd_fpn_ssld_2x_coco.pdparams) |
| SOLOv2            | 轻量级单阶段实例分割算法 | 云边端  | mask AP: 38.0                    | [链接](configs/solov2/solov2_r50_fpn_3x_coco.yml)                       | [下载地址](https://paddledet.bj.bcebos.com/models/solov2_r50_fpn_3x_coco.pdparams)                    |

</details>

<details>
<summary><b> 3. 关键点检测</b></summary>

| 模型名称                                        | 模型简介                                                             | 推荐场景                               | COCO精度（AP） | 速度                      | 配置文件                                                    | 模型下载                                                                                    |
|:------------------------------------------- |:---------------------------------------------------------------- |:---------------------------------- |:----------:|:-----------------------:|:-------------------------------------------------------:|:---------------------------------------------------------------------------------------:|
| HRNet-w32 + DarkPose                        | <div style="width: 130pt">top-down 关键点检测算法<br/>输入尺寸384x288</div> | <div style="width: 50pt">云边端</div> | 78.3       | T4 TensorRT FP16 2.96ms | [链接](configs/keypoint/hrnet/dark_hrnet_w32_384x288.yml) | [下载地址](https://paddledet.bj.bcebos.com/models/keypoint/dark_hrnet_w32_384x288.pdparams) |
| HRNet-w32 + DarkPose                        | top-down 关键点检测算法<br/>输入尺寸256x192                                 | 云边端                                | 78.0       | T4 TensorRT FP16 1.75ms | [链接](configs/keypoint/hrnet/dark_hrnet_w32_256x192.yml) | [下载地址](https://paddledet.bj.bcebos.com/models/keypoint/dark_hrnet_w32_256x192.pdparams) |
| [PP-TinyPose](./configs/keypoint/tiny_pose) | 轻量级关键点算法<br/>输入尺寸256x192                                         | 移动端                                | 68.8       | 骁龙865 四线程 6.30ms        | [链接](configs/keypoint/tiny_pose/tinypose_256x192.yml)   | [下载地址](https://bj.bcebos.com/v1/paddledet/models/keypoint/tinypose_256x192.pdparams)    |
| [PP-TinyPose](./configs/keypoint/tiny_pose) | 轻量级关键点算法<br/>输入尺寸128x96                                          | 移动端                                | 58.1       | 骁龙865 四线程 2.37ms        | [链接](configs/keypoint/tiny_pose/tinypose_128x96.yml)    | [下载地址](https://bj.bcebos.com/v1/paddledet/models/keypoint/tinypose_128x96.pdparams)     |

#### 其他关键点检测模型 [文档链接](configs/keypoint)

</details>

<details>
<summary><b> 4. 多目标跟踪PP-Tracking </b></summary>

| 模型名称      | 模型简介                     | 推荐场景                               | 精度                     | 配置文件                                                                  | 模型下载                                                                                              |
|:--------- |:------------------------ |:---------------------------------- |:----------------------:|:---------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------:|
| DeepSORT  | SDE多目标跟踪算法 检测、ReID模型相互独立 | <div style="width: 50pt">云边端</div> | MOT-17 half val:  66.9 | [链接](configs/mot/deepsort/deepsort_jde_yolov3_pcb_pyramid.yml)        | [下载地址](https://paddledet.bj.bcebos.com/models/mot/deepsort/deepsort_pcb_pyramid_r101.pdparams)    |
| ByteTrack | SDE多目标跟踪算法 仅包含检测模型       | 云边端                                | MOT-17 half val:  77.3 | [链接](configs/mot/bytetrack/detector/yolox_x_24e_800x1440_mix_det.yml) | [下载地址](https://paddledet.bj.bcebos.com/models/mot/deepsort/yolox_x_24e_800x1440_mix_det.pdparams) |
| JDE       | JDE多目标跟踪算法 多任务联合学习方法     | 云边端                                | MOT-16 test: 64.6      | [链接](configs/mot/jde/jde_darknet53_30e_1088x608.yml)                  | [下载地址](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_1088x608.pdparams)            |
| FairMOT   | JDE多目标跟踪算法 多任务联合学习方法     | 云边端                                | MOT-16 test: 75.0      | [链接](configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml)              | [下载地址](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608.pdparams)            |

#### 其他多目标跟踪模型 [文档链接](configs/mot)

</details>

<details>
<summary><b> 5. 产业级实时行人分析工具</b></summary>


| 任务            | 端到端速度（ms）|  模型方案  |  模型体积 |
| :---------:     | :-------:  |  :------: |:------: |
|  行人检测（高精度）  | 25.1ms  |  [目标检测](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip) | 182M |  
|  行人检测（轻量级）  | 16.2ms  |  [目标检测](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_s_36e_pipeline.zip) | 27M |
|  行人跟踪（高精度）  | 31.8ms  |  [多目标跟踪](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip) | 182M |  
|  行人跟踪（轻量级）  | 21.0ms  |  [多目标跟踪](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_s_36e_pipeline.zip) | 27M |
|  属性识别（高精度）  |   单人8.5ms | [目标检测](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)<br> [属性识别](https://bj.bcebos.com/v1/paddledet/models/pipeline/PPHGNet_small_person_attribute_954_infer.zip) | 目标检测：182M<br>属性识别：86M |
|  属性识别（轻量级）  |   单人7.1ms | [目标检测](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)<br> [属性识别](https://bj.bcebos.com/v1/paddledet/models/pipeline/PPLCNet_x1_0_person_attribute_945_infer.zip) | 目标检测：182M<br>属性识别：86M |
|  摔倒识别  |   单人10ms | [多目标跟踪](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip) <br> [关键点检测](https://bj.bcebos.com/v1/paddledet/models/pipeline/dark_hrnet_w32_256x192.zip) <br> [基于关键点行为识别](https://bj.bcebos.com/v1/paddledet/models/pipeline/STGCN.zip) | 多目标跟踪：182M<br>关键点检测：101M<br>基于关键点行为识别：21.8M |
|  闯入识别  |   31.8ms | [多目标跟踪](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip) | 182M |
|  打架识别  |   19.7ms | [视频分类](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip) | 90M |
|  抽烟识别  |   单人15.1ms | [目标检测](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)<br>[基于人体id的目标检测](https://bj.bcebos.com/v1/paddledet/models/pipeline/ppyoloe_crn_s_80e_smoking_visdrone.zip) | 目标检测：182M<br>基于人体id的目标检测：27M |
|  打电话识别  |   单人ms | [目标检测](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)<br>[基于人体id的图像分类](https://bj.bcebos.com/v1/paddledet/models/pipeline/PPHGNet_tiny_calling_halfbody.zip) | 目标检测：182M<br>基于人体id的图像分类：45M |


点击模型方案中的模型即可下载指定模型

详细信息参考[文档](deploy/pipeline)

</details>


## <img src="https://user-images.githubusercontent.com/48054808/157828296-d5eb0ccb-23ea-40f5-9957-29853d7d13a9.png" width="20"/> 文档教程

### 入门教程

- [安装说明](docs/tutorials/INSTALL_cn.md)
- [快速体验](docs/tutorials/QUICK_STARTED_cn.md)
- [数据准备](docs/tutorials/data/README.md)
- [PaddleDetection全流程使用](docs/tutorials/GETTING_STARTED_cn.md)
- [自定义数据训练](docs/tutorials/CustomizeDataTraining.md)
- [FAQ/常见问题汇总](docs/tutorials/FAQ)

### 进阶教程

- 参数配置

  - [RCNN参数说明](docs/tutorials/config_annotation/faster_rcnn_r50_fpn_1x_coco_annotation.md)
  - [PP-YOLO参数说明](docs/tutorials/config_annotation/ppyolo_r50vd_dcn_1x_coco_annotation.md)

- 模型压缩(基于[PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim))

  - [剪裁/量化/蒸馏教程](configs/slim)

- [推理部署](deploy/README.md)

  - [模型导出教程](deploy/EXPORT_MODEL.md)
  - [Paddle Inference部署](deploy/README.md)
    - [Python端推理部署](deploy/python)
    - [C++端推理部署](deploy/cpp)
  - [Paddle-Lite部署](deploy/lite)
  - [Paddle Serving部署](deploy/serving)
  - [ONNX模型导出](deploy/EXPORT_ONNX_MODEL.md)
  - [推理benchmark](deploy/BENCHMARK_INFER.md)

- 进阶开发

  - [数据处理模块](docs/advanced_tutorials/READER.md)
  - [新增检测模型](docs/advanced_tutorials/MODEL_TECHNICAL.md)
  - 二次开发教程
    - [目标检测](docs/advanced_tutorials/customization/detection.md)
    - [关键点检测](docs/advanced_tutorials/customization/keypoint_detection.md)
    - [多目标跟踪](docs/advanced_tutorials/customization/mot.md)
    - [行为识别](docs/advanced_tutorials/customization/action.md)
    - [属性识别](docs/advanced_tutorials/customization/attribute.md)

### 课程专栏

- **【理论基础】[目标检测7日打卡营](https://aistudio.baidu.com/aistudio/education/group/info/1617)：** 目标检测任务综述、RCNN系列目标检测算法详解、YOLO系列目标检测算法详解、PP-YOLO优化策略与案例分享、AnchorFree系列算法介绍和实践

- **【产业实践】[AI快车道产业级目标检测技术与应用](https://aistudio.baidu.com/aistudio/education/group/info/23670)：** 目标检测超强目标检测算法矩阵、实时行人分析系统PP-Human、目标检测产业应用全流程拆解与实践

- **【行业特色】2022.3.26 [智慧城市行业七日课](https://aistudio.baidu.com/aistudio/education/group/info/25620)：** 城市规划、城市治理、智慧政务、交通管理、社区治理

### [产业实践范例教程](./industrial_tutorial/README.md)

- [基于PP-PicoDet增强版的路面垃圾检测](https://aistudio.baidu.com/aistudio/projectdetail/3846170?channelType=0&channel=0)

- [基于PP-PicoDet的通信塔识别及Android端部署](https://aistudio.baidu.com/aistudio/projectdetail/3561097)

- [基于Faster-RCNN的瓷砖表面瑕疵检测](https://aistudio.baidu.com/aistudio/projectdetail/2571419)

- [基于PaddleDetection的PCB瑕疵检测](https://aistudio.baidu.com/aistudio/projectdetail/2367089)

- [基于FairMOT实现人流量统计](https://aistudio.baidu.com/aistudio/projectdetail/2421822)

- [基于YOLOv3实现跌倒检测 ](https://aistudio.baidu.com/aistudio/projectdetail/2500639)

- [基于人体关键点检测的合规检测](https://aistudio.baidu.com/aistudio/projectdetail/4061642?contributionType=1)

- [打架识别](https://aistudio.baidu.com/aistudio/projectdetail/4086987?contributionType=1)

## <img title="" src="https://user-images.githubusercontent.com/48054808/157836473-1cf451fa-f01f-4148-ba68-b6d06d5da2f9.png" alt="" width="20"> 应用案例

- [安卓健身APP](https://github.com/zhiboniu/pose_demo_android)
- [多目标跟踪系统GUI可视化界面](https://github.com/yangyudong2020/PP-Tracking_GUi)

## <img src="https://user-images.githubusercontent.com/48054808/160552806-496dc3ba-beb6-4623-8e26-44416b5848bf.png" width="25"/> 第三方教程推荐

- [PaddleDetection在Windows下的部署(一)](https://zhuanlan.zhihu.com/p/268657833)
- [PaddleDetection在Windows下的部署(二)](https://zhuanlan.zhihu.com/p/280206376)
- [Jetson Nano上部署PaddleDetection经验分享](https://zhuanlan.zhihu.com/p/319371293)
- [安全帽检测YOLOv3模型在树莓派上的部署](https://github.com/PaddleCV-FAQ/PaddleDetection-FAQ/blob/main/Lite%E9%83%A8%E7%BD%B2/yolov3_for_raspi.md)
- [使用SSD-MobileNetv1完成一个项目--准备数据集到完成树莓派部署](https://github.com/PaddleCV-FAQ/PaddleDetection-FAQ/blob/main/Lite%E9%83%A8%E7%BD%B2/ssd_mobilenet_v1_for_raspi.md)

## <img src="https://user-images.githubusercontent.com/48054808/157835981-ef6057b4-6347-4768-8fcc-cd07fcc3d8b0.png" width="20"/> 版本更新

版本更新内容请参考[版本更新文档](docs/CHANGELOG.md)


## <img src="https://user-images.githubusercontent.com/48054808/157835796-08d4ffbc-87d9-4622-89d8-cf11a44260fc.png" width="20"/> 贡献代码

我们非常欢迎你可以为PaddleDetection提供代码，也十分感谢你的反馈。

- 感谢[Mandroide](https://github.com/Mandroide)清理代码并且统一部分函数接口。
- 感谢[FL77N](https://github.com/FL77N/)贡献`Sparse-RCNN`模型。
- 感谢[Chen-Song](https://github.com/Chen-Song)贡献`Swin Faster-RCNN`模型。
- 感谢[yangyudong](https://github.com/yangyudong2020), [hchhtc123](https://github.com/hchhtc123) 开发PP-Tracking GUI界面
- 感谢Shigure19 开发PP-TinyPose健身APP
- 感谢[manangoel99](https://github.com/manangoel99)贡献Wandb可视化方式

## <img src="https://user-images.githubusercontent.com/48054808/157835276-9aab9d1c-1c46-446b-bdd4-5ab75c5cfa48.png" width="20"/> 引用

```
@misc{ppdet2019,
title={PaddleDetection, Object detection and instance segmentation toolkit based on PaddlePaddle.},
author={PaddlePaddle Authors},
howpublished = {\url{https://github.com/PaddlePaddle/PaddleDetection}},
year={2019}
}
```
