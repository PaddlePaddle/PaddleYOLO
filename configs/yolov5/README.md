# YOLOv5

## 内容
- [模型库](#模型库)
- [使用说明](#使用说明)
- [速度测试](#速度测试)

## 模型库

### 基础模型

| 网络网络        | 输入尺寸   | 图片数/GPU | 学习率策略 | 模型推理耗时(ms) | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | Params(M) | FLOPs(G) |    下载链接       | 配置文件 |
| :------------- | :------- | :-------: | :------: | :------------: | :---------------------: | :----------------: |:---------: | :------: |:---------------: |:-----: |
| YOLOv5-n        |  640     |    16     |   300e    |     1.5    |  28.0  | 45.7 |  1.87  | 4.52 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov5_n_300e_coco.pdparams) | [配置文件](./yolov5_n_300e_coco.yml) |
| YOLOv5-s        |  640     |    16      |   300e   |     2.6    |  37.6  | 56.7 |  7.24  | 16.54 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov5_s_300e_coco.pdparams) | [配置文件](./yolov5_s_300e_coco.yml) |
| YOLOv5-m        |  640     |    16      |   300e   |     5.2    |  45.4  | 64.1 |  21.19  | 49.08 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov5_m_300e_coco.pdparams) | [配置文件](./yolov5_m_300e_coco.yml) |
| YOLOv5-l        |  640     |    16      |   300e   |     7.9    |  48.9  | 67.1 |  46.56  | 109.32 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov5_l_300e_coco.pdparams) | [配置文件](./yolov5_l_300e_coco.yml) |
| YOLOv5-x        |  640     |    16      |   300e   |     13.7   |  50.6  | 68.7 |  86.75  | 205.92 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov5_x_300e_coco.pdparams) | [配置文件](./yolov5_x_300e_coco.yml) |
| YOLOv5-s ConvNeXt|  640    |    8       |   36e    |      -     |  42.4  | 65.3 |  34.54  |  17.96 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov5_convnext_s_36e_coco.pdparams) | [配置文件](../convnext/yolov5_convnext_s_36e_coco.yml) |

### SSOD预训练模型

| 网络网络        | 输入尺寸   | 图片数/GPU | 学习率策略 | 模型推理耗时(ms) | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | Params(M) | FLOPs(G) |    下载链接       | 配置文件 |
| :------------- | :------- | :-------: | :------: | :------------: | :---------------------: | :----------------: |:---------: | :------: |:---------------: |:-----: |
| YOLOv5-s        |  640     |    16      |   80e   |     2.6    |  38.8  | 58.2 |  7.24  | 16.54 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov5_s_80e_ssod_finetune_coco.pdparams) | [配置文件](./yolov5_s_80e_ssod_finetune_coco.yml) |

### Objects-365预训练模型

| 网络网络        | 输入尺寸   | 图片数/GPU | 学习率策略 | 模型推理耗时(ms) | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | Params(M) | FLOPs(G) |    下载链接       | 配置文件 |
| :------------- | :------- | :-------: | :------: | :------------: | :---------------------: | :----------------: |:---------: | :------: |:---------------: |:-----: |
| YOLOv5-l        |  640     |    16      |   30e   |     7.9    |  49.3  | 67.8 |  46.56  | 109.32 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov5_l_30e_obj365_finetune_coco.pdparams) | [配置文件](./yolov5_l_30e_obj365_finetune_coco.yml) |

### P6大尺度模型

| 网络网络        | 输入尺寸   | 图片数/GPU | 学习率策略 | 模型推理耗时(ms) | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | Params(M) | FLOPs(G) |    下载链接       | 配置文件 |
| :------------- | :------- | :-------: | :------: | :------------: | :---------------------: | :----------------: |:---------: | :------: |:---------------: |:-----: |
| YOLOv5p6-n        |  1280     |    16     |   300e    |     -    |  35.9  | 54.2 |  3.25  | 9.23 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov5p6_n_300e_coco.pdparams) | [配置文件](./yolov5p6_n_300e_coco.yml) |
| YOLOv5p6-s        |  1280     |    16     |   300e    |     -    |  44.5  | 63.3 |  12.63  | 33.81 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov5p6_s_300e_coco.pdparams) | [配置文件](./yolov5p6_s_300e_coco.yml) |
| YOLOv5p6-m        |  1280     |    16     |   300e    |     -    |  51.1  | 69.0 |  35.73  | 100.21 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov5p6_m_300e_coco.pdparams) | [配置文件](./yolov5p6_m_300e_coco.yml) |
| YOLOv5p6-l        |  1280     |    8      |   300e    |     -    |  53.4  | 71.0 |  76.77  | 223.09 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov5p6_l_300e_coco.pdparams) | [配置文件](./yolov5p6_l_300e_coco.yml) |
| YOLOv5p6-x        |  1280     |    8      |   300e    |     -    |  54.7  | 72.4 |  140.80 | 420.03 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov5p6_x_300e_coco.pdparams) | [配置文件](./yolov5p6_x_300e_coco.yml) |

### [YOLOv5u](../yolov5u)

| 网络网络        | 输入尺寸   | 图片数/GPU | 学习率策略 | 模型推理耗时(ms) | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | Params(M) | FLOPs(G) |    下载链接       | 配置文件 |
| :------------- | :------- | :-------: | :------: | :------------: | :---------------------: | :----------------: |:---------: | :------: |:---------------: |:-----: |
| YOLOv5u-n        |  640     |    16      |   300e   |     1.61    |  34.5  | 49.7 |  2.65  | 7.79 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov5u_n_300e_coco.pdparams) | [配置文件](./yolov5u/yolov5u_n_300e_coco.yml) |
| YOLOv5u-s        |  640     |    16      |   300e   |     2.66    |  43.0  | 59.7 |  9.15   | 24.12 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov5u_s_300e_coco.pdparams) | [配置文件](./yolov5u/yolov5u_s_300e_coco.yml) |
| YOLOv5u-m        |  640     |    16      |   300e   |     5.50    |  49.0  | 65.7 |  25.11  | 64.42 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov5u_m_300e_coco.pdparams) | [配置文件](./yolov5u/yolov5u_m_300e_coco.yml) |
| YOLOv5u-l        |  640     |    16      |   300e   |     8.73    |  52.2  | 69.0 |  53.23  | 135.34 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov5u_l_300e_coco.pdparams) | [配置文件](./yolov5u/yolov5u_l_300e_coco.yml) |
| YOLOv5u-x        |  640     |    16      |   300e   |     15.49   |  53.1  | 69.9 |  97.28  | 246.89 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov5u_x_300e_coco.pdparams) | [配置文件](./yolov5u/yolov5u_x_300e_coco.yml) |


**注意:**
  - YOLOv5模型训练使用COCO train2017作为训练集，Box AP为在COCO val2017上的`mAP(IoU=0.5:0.95)`结果；
  - YOLOv5u 模型表示YOLOv5结构使用YOLOv8的head和loss，是Anchor Free的检测方案，具体可参照[YOLOv5u](../yolov5u)；
  - YOLOv5模型训练过程中默认使用8 GPUs进行混合精度训练，默认lr为0.01为8卡总batch_size的设置，如果**GPU卡数**或者每卡**batch size**发生改动，也不需要改动学习率，但为了保证高精度最好使用**总batch size大于64**的配置去训练；
  - 模型推理耗时(ms)为TensorRT-FP16下测试的耗时，不包含数据预处理和模型输出后处理(NMS)的耗时。测试采用单卡Tesla T4 GPU，batch size=1，测试环境为**paddlepaddle-2.3.2**, **CUDA 11.2**, **CUDNN 8.2**, **GCC-8.2**, **TensorRT 8.0.3.4**，具体请参考[速度测试](#速度测试)。
  - 如果你设置了`--run_benchmark=True`, 你首先需要安装以下依赖`pip install pynvml psutil GPUtil`。

### 部署模型

| 网络模型     | 输入尺寸 | 导出后的权重(w/o NMS) | ONNX(w/o NMS)  |
| :-------- | :--------: | :---------------------: | :----------------: |
| YOLOv5-n |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_n_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_n_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_n_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_n_300e_coco_wo_nms.onnx) |
| YOLOv5-s |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_s_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_s_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_s_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_s_300e_coco_wo_nms.onnx) |
| YOLOv5-m |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_m_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_m_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_m_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_m_300e_coco_wo_nms.onnx) |
| YOLOv5-l |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_l_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_l_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_l_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_l_300e_coco_wo_nms.onnx) |
| YOLOv5-x |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_x_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_x_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_x_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/yolov5/yolov5_x_300e_coco_wo_nms.onnx) |

## 使用教程

### 0. **一键运行全流程**

将以下命令写在一个脚本文件里如```run.sh```，一键运行命令为：```sh run.sh```，也可命令行一句句去运行。

```bash
model_name=yolov5 # 可修改，如 ppyoloe
job_name=yolov5_s_300e_coco # 可修改，如 ppyoloe_plus_crn_s_80e_coco

config=configs/${model_name}/${job_name}.yml
log_dir=log_dir/${job_name}
# weights=https://bj.bcebos.com/v1/paddledet/models/${job_name}.pdparams
weights=output/${job_name}/model_final.pdparams

# 1.训练（单卡/多卡），加 --eval 表示边训边评估，加 --amp 表示混合精度训练
# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c ${config} --eval --amp
python -m paddle.distributed.launch --log_dir=${log_dir} --gpus 0,1,2,3,4,5,6,7 tools/train.py -c ${config} --eval --amp

# 2.评估，加 --classwise 表示输出每一类mAP
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c ${config} -o weights=${weights} --classwise

# 3.预测 (单张图/图片文件夹）
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c ${config} -o weights=${weights} --infer_img=demo/000000014439_640x640.jpg --draw_threshold=0.5
# CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c ${config} -o weights=${weights} --infer_dir=demo/ --draw_threshold=0.5

# 4.导出模型，以下3种模式选一种
## 普通导出，加trt表示用于trt加速，对NMS和silu激活函数提速明显
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c ${config} -o weights=${weights} # trt=True

## exclude_post_process去除后处理导出，返回和YOLOv5导出ONNX时相同格式的concat后的1个Tensor，是未缩放回原图的坐标+分类置信度
# CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c ${config} -o weights=${weights} exclude_post_process=True # trt=True

## exclude_nms去除NMS导出，返回2个Tensor，是缩放回原图后的坐标和分类置信度
# CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c ${config} -o weights=${weights} exclude_nms=True # trt=True

# 5.部署预测，注意不能使用 去除后处理 或 去除NMS 导出后的模型去预测
CUDA_VISIBLE_DEVICES=0 python deploy/python/infer.py --model_dir=output_inference/${job_name} --image_file=demo/000000014439_640x640.jpg --device=GPU

# 6.部署测速，加 “--run_mode=trt_fp16” 表示在TensorRT FP16模式下测速，注意如需用到 trt_fp16 则必须为加 trt=True 导出的模型
CUDA_VISIBLE_DEVICES=0 python deploy/python/infer.py --model_dir=output_inference/${job_name} --image_file=demo/000000014439_640x640.jpg --device=GPU --run_benchmark=True # --run_mode=trt_fp16

# 7.onnx导出，一般结合 exclude_post_process去除后处理导出的模型
paddle2onnx --model_dir output_inference/${job_name} --model_filename model.pdmodel --params_filename model.pdiparams --opset_version 12 --save_file ${job_name}.onnx

# 8.onnx trt测速
/usr/local/TensorRT-8.0.3.4/bin/trtexec --onnx=${job_name}.onnx --workspace=4096 --avgRuns=10 --shapes=input:1x3x640x640 --fp16
/usr/local/TensorRT-8.0.3.4/bin/trtexec --onnx=${job_name}.onnx --workspace=4096 --avgRuns=10 --shapes=input:1x3x640x640 --fp32
```

### 1. 训练
执行以下指令使用混合精度训练YOLOv5
```bash
python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/yolov5/yolov5_s_300e_coco.yml --amp --eval
```
**注意:**
- `--amp`表示开启混合精度训练以避免显存溢出，`--eval`表示边训边验证。

### 2. 评估
执行以下命令在单个GPU上评估COCO val2017数据集
```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/yolov5/yolov5_s_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/yolov5_s_300e_coco.pdparams
```

### 3. 推理
使用以下命令在单张GPU上预测图片，使用`--infer_img`推理单张图片以及使用`--infer_dir`推理文件中的所有图片。
```bash
# 推理单张图片
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/yolov5/yolov5_s_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/yolov5_s_300e_coco.pdparams --infer_img=demo/000000014439_640x640.jpg

# 推理文件中的所有图片
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/yolov5/yolov5_s_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/yolov5_s_300e_coco.pdparams --infer_dir=demo
```

### 4.导出模型
YOLOv5在GPU上推理部署或benchmark测速等需要通过`tools/export_model.py`导出模型。

当你**使用Paddle Inference但不使用TensorRT**时，运行以下的命令导出模型

```bash
python tools/export_model.py -c configs/yolov5/yolov5_s_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/yolov5_s_300e_coco.pdparams
```

当你**使用Paddle Inference且使用TensorRT**时，需要指定`-o trt=True`来导出模型。

```bash
python tools/export_model.py -c configs/yolov5/yolov5_s_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/yolov5_s_300e_coco.pdparams trt=True
```

如果你想将YOLOv5模型导出为**ONNX格式**，参考
[PaddleDetection模型导出为ONNX格式教程](../../deploy/EXPORT_ONNX_MODEL.md)，运行以下命令：

```bash

# 导出推理模型
python tools/export_model.py -c configs/yolov5/yolov5_s_300e_coco.yml --output_dir=output_inference -o weights=https://paddledet.bj.bcebos.com/models/yolov5_s_300e_coco.pdparams

# 安装paddle2onnx
pip install paddle2onnx

# 转换成onnx格式
paddle2onnx --model_dir output_inference/yolov5_s_300e_coco --model_filename model.pdmodel --params_filename model.pdiparams --opset_version 11 --save_file yolov5_s_300e_coco.onnx
```

**注意：** ONNX模型目前只支持batch_size=1


### 5.推理部署
YOLOv5可以使用以下方式进行部署：
  - Paddle Inference [Python](../../deploy/python) & [C++](../../deploy/cpp)
  - [Paddle-TensorRT](../../deploy/TENSOR_RT.md)
  - [PaddleServing](https://github.com/PaddlePaddle/Serving)
  - [PaddleSlim模型量化](../slim)

运行以下命令导出模型

```bash
python tools/export_model.py -c configs/yolov5/yolov5_s_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/yolov5_s_300e_coco.pdparams trt=True
```

**注意：**
- trt=True表示**使用Paddle Inference且使用TensorRT**进行测速，速度会更快，默认不加即为False，表示**使用Paddle Inference但不使用TensorRT**进行测速。
- 如果是使用Paddle Inference在TensorRT FP16模式下部署，需要参考[Paddle Inference文档](https://www.paddlepaddle.org.cn/inference/master/user_guides/download_lib.html#python)，下载并安装与你的CUDA, CUDNN和TensorRT相应的wheel包。

#### 5.1.Python部署
`deploy/python/infer.py`使用上述导出后的Paddle Inference模型用于推理和benchnark测速，如果设置了`--run_benchmark=True`, 首先需要安装以下依赖`pip install pynvml psutil GPUtil`。

```bash
# Python部署推理单张图片
python deploy/python/infer.py --model_dir=output_inference/yolov5_s_300e_coco --image_file=demo/000000014439_640x640.jpg --device=gpu

# 推理文件夹下的所有图片
python deploy/python/infer.py --model_dir=output_inference/yolov5_s_300e_coco --image_dir=demo/ --device=gpu
```

#### 5.2. C++部署
`deploy/cpp/build/main`使用上述导出后的Paddle Inference模型用于C++推理部署, 首先按照[docs](../../deploy/cpp/docs)编译安装环境。
```bash
# C++部署推理单张图片
./deploy/cpp/build/main --model_dir=output_inference/yolov5_s_300e_coco/ --image_file=demo/000000014439_640x640.jpg --run_mode=paddle --device=GPU --threshold=0.5 --output_dir=cpp_infer_output/yolov5_s_300e_coco
```


## 速度测试

为了公平起见，在[模型库](#模型库)中的速度测试结果均为不包含数据预处理和模型输出后处理(NMS)的数据(与[YOLOv4(AlexyAB)](https://github.com/AlexeyAB/darknet)测试方法一致)，需要在导出模型时指定`-o exclude_nms=True`。测速需设置`--run_benchmark=True`, 首先需要安装以下依赖`pip install pynvml psutil GPUtil`。

**使用Paddle Inference但不使用TensorRT**进行测速，执行以下命令：

```bash
# 导出模型
python tools/export_model.py -c configs/yolov5/yolov5_s_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/yolov5_s_300e_coco.pdparams exclude_nms=True

# 速度测试，使用run_benchmark=True
python deploy/python/infer.py --model_dir=output_inference/yolov5_s_300e_coco --image_file=demo/000000014439_640x640.jpg --run_mode=paddle --device=gpu --run_benchmark=True
```

**使用Paddle Inference且使用TensorRT**进行测速，执行以下命令：

```bash
# 导出模型，使用trt=True
python tools/export_model.py -c configs/yolov5/yolov5_s_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/yolov5_s_300e_coco.pdparams exclude_nms=True trt=True

# 速度测试，使用run_benchmark=True
python deploy/python/infer.py --model_dir=output_inference/yolov5_s_300e_coco --image_file=demo/000000014439_640x640.jpg --device=gpu --run_benchmark=True

# tensorRT-FP32测速
python deploy/python/infer.py --model_dir=output_inference/yolov5_s_300e_coco --image_file=demo/000000014439_640x640.jpg --device=gpu --run_benchmark=True --run_mode=trt_fp32

# tensorRT-FP16测速
python deploy/python/infer.py --model_dir=output_inference/yolov5_s_300e_coco --image_file=demo/000000014439_640x640.jpg --device=gpu --run_benchmark=True --run_mode=trt_fp16
```
**注意:**
- 导出模型时指定`-o exclude_nms=True`仅作为测速时用，这样导出的模型其推理部署预测的结果不是最终检出框的结果。
- [模型库](#模型库)中的速度测试结果为tensorRT-FP16测速后的最快速度，为不包含数据预处理和模型输出后处理(NMS)的耗时。
