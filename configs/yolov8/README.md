# YOLOv8

## 内容
- [模型库](#模型库)
- [使用说明](#使用说明)
- [速度测试](#速度测试)
- [引用](#引用)

## 模型库

### 基础模型

| 网络网络        | 输入尺寸   | 图片数/GPU | 学习率策略 | TRT-FP16-Latency(ms) | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | Params(M) | FLOPs(G) |    下载链接       | 配置文件 |
| :------------- | :------- | :-------: | :------: | :------------: | :---------------------: | :----------------: |:---------: | :------: |:---------------: |:-----: |
| *YOLOv8-n        |  640     |    16      |   500e   |    2.4   |  37.3  | 53.0 |  3.16   | 8.7 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov8_n_500e_coco.pdparams) | [配置文件](./yolov8_n_500e_coco.yml) |
| *YOLOv8-s        |  640     |    16      |   500e   |    3.4   |  44.9  | 61.8 |  11.17  | 28.6 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov8_s_500e_coco.pdparams) | [配置文件](./yolov8_s_500e_coco.yml) |
| *YOLOv8-m        |  640     |    16      |   500e   |    6.5   |  50.2  | 67.3 |  25.90  | 78.9 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov8_m_500e_coco.pdparams) | [配置文件](./yolov8_m_500e_coco.yml) |
| *YOLOv8-l        |  640     |    16      |   500e   |    10.0  |  52.8  | 69.6 |  43.69  | 165.2 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov8_l_500e_coco.pdparams) | [配置文件](./yolov8_l_500e_coco.yml) |
| *YOLOv8-x        |  640     |    16      |   500e   |    15.1  |  53.8  | 70.6 |  68.23  | 257.8 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov8_x_500e_coco.pdparams) | [配置文件](./yolov8_x_500e_coco.yml) |

### P6大尺度模型

| 网络网络        | 输入尺寸   | 图片数/GPU | 学习率策略 | TRT-FP16-Latency(ms) | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | Params(M) | FLOPs(G) |    下载链接       | 配置文件 |
| :------------- | :------- | :-------: | :------: | :------------: | :---------------------: | :----------------: |:---------: | :------: |:---------------: |:-----: |
| *YOLOv8-P6-x     |  1280    |    16      |   500e   |    -  |  -  | - |  97.42  | 522.93 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov8p6_x_500e_coco.pdparams) | [配置文件](./yolov8p6_x_500e_coco.yml) |


**注意:**
  - YOLOv8模型暂未支持完全训练，mAP为部署权重在COCO val2017上的`mAP(IoU=0.5:0.95)`结果，且评估未使用`multi_label`等trick；
  - YOLOv8模型训练使用COCO train2017作为训练集，Box AP为在COCO val2017上的`mAP(IoU=0.5:0.95)`结果；
  - YOLOv8模型训练过程中默认使用8 GPUs进行混合精度训练，默认lr为0.01为8卡总batch_size的设置，如果**GPU卡数**或者每卡**batch size**发生改动，也不需要改动学习率，但为了保证高精度最好使用**总batch size大于64**的配置去训练；
  - TRT-FP16-Latency(ms)模型推理耗时为TensorRT-FP16下测试的耗时，不包含数据预处理和模型输出后处理(NMS)的耗时。测试采用**单卡Tesla T4 GPU**，batch size=1，测试环境为**paddlepaddle-2.3.2**, **CUDA 11.2**, **CUDNN 8.2**, **GCC-8.2**, **TensorRT 8.0.3.4**，具体请参考[速度测试](#速度测试)。
  - 如果你设置了`--run_benchmark=True`, 你首先需要安装以下依赖`pip install pynvml psutil GPUtil`。

### 部署模型



## 使用教程

### 1. 训练
执行以下指令使用混合精度训练YOLOv8
```bash
python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/yolov8/yolov8_s_500e_coco.yml --amp --eval
```
**注意:**
- `--amp`表示开启混合精度训练以避免显存溢出，`--eval`表示边训边验证。

### 2. 评估
执行以下命令在单个GPU上评估COCO val2017数据集
```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/yolov8/yolov8_s_500e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/yolov8_s_500e_coco.pdparams
```

### 3. 推理
使用以下命令在单张GPU上预测图片，使用`--infer_img`推理单张图片以及使用`--infer_dir`推理文件中的所有图片。
```bash
# 推理单张图片
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/yolov8/yolov8_s_500e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/yolov8_s_500e_coco.pdparams --infer_img=demo/000000014439_640x640.jpg

# 推理文件中的所有图片
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/yolov8/yolov8_s_500e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/yolov8_s_500e_coco.pdparams --infer_dir=demo
```

### 4.导出模型
YOLOv8在GPU上推理部署或benchmark测速等需要通过`tools/export_model.py`导出模型。

当你**使用Paddle Inference但不使用TensorRT**时，运行以下的命令导出模型

```bash
python tools/export_model.py -c configs/yolov8/yolov8_s_500e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/yolov8_s_500e_coco.pdparams
```

当你**使用Paddle Inference且使用TensorRT**时，需要指定`-o trt=True`来导出模型。

```bash
python tools/export_model.py -c configs/yolov8/yolov8_s_500e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/yolov8_s_500e_coco.pdparams trt=True
```

如果你想将YOLOv8模型导出为**ONNX格式**，参考
[PaddleDetection模型导出为ONNX格式教程](../../deploy/EXPORT_ONNX_MODEL.md)，运行以下命令：

```bash

# 导出推理模型
python tools/export_model.py -c configs/yolov8/yolov8_s_500e_coco.yml --output_dir=output_inference -o weights=https://paddledet.bj.bcebos.com/models/yolov8_s_500e_coco.pdparams

# 安装paddle2onnx
pip install paddle2onnx

# 转换成onnx格式
paddle2onnx --model_dir output_inference/yolov8_s_500e_coco --model_filename model.pdmodel --params_filename model.pdiparams --opset_version 11 --save_file yolov8_s_500e_coco.onnx
```

**注意：** ONNX模型目前只支持batch_size=1


### 5.推理部署
YOLOv8可以使用以下方式进行部署：
  - Paddle Inference [Python](../../deploy/python) & [C++](../../deploy/cpp)
  - [Paddle-TensorRT](../../deploy/TENSOR_RT.md)
  - [PaddleServing](https://github.com/PaddlePaddle/Serving)
  - [PaddleSlim模型量化](../slim)

运行以下命令导出模型

```bash
python tools/export_model.py -c configs/yolov8/yolov8_s_500e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/yolov8_s_500e_coco.pdparams trt=True
```

**注意：**
- trt=True表示**使用Paddle Inference且使用TensorRT**进行测速，速度会更快，默认不加即为False，表示**使用Paddle Inference但不使用TensorRT**进行测速。
- 如果是使用Paddle Inference在TensorRT FP16模式下部署，需要参考[Paddle Inference文档](https://www.paddlepaddle.org.cn/inference/master/user_guides/download_lib.html#python)，下载并安装与你的CUDA, CUDNN和TensorRT相应的wheel包。

#### 5.1.Python部署
`deploy/python/infer.py`使用上述导出后的Paddle Inference模型用于推理和benchnark测速，如果设置了`--run_benchmark=True`, 首先需要安装以下依赖`pip install pynvml psutil GPUtil`。

```bash
# Python部署推理单张图片
python deploy/python/infer.py --model_dir=output_inference/yolov8_s_500e_coco --image_file=demo/000000014439_640x640.jpg --device=gpu

# 推理文件夹下的所有图片
python deploy/python/infer.py --model_dir=output_inference/yolov8_s_500e_coco --image_dir=demo/ --device=gpu
```

#### 5.2. C++部署
`deploy/cpp/build/main`使用上述导出后的Paddle Inference模型用于C++推理部署, 首先按照[docs](../../deploy/cpp/docs)编译安装环境。
```bash
# C++部署推理单张图片
./deploy/cpp/build/main --model_dir=output_inference/yolov8_s_500e_coco/ --image_file=demo/000000014439_640x640.jpg --run_mode=paddle --device=GPU --threshold=0.5 --output_dir=cpp_infer_output/yolov8_s_500e_coco
```


## 速度测试

为了公平起见，在[模型库](#模型库)中的速度测试结果均为不包含数据预处理和模型输出后处理(NMS)的数据(与[YOLOv4(AlexyAB)](https://github.com/AlexeyAB/darknet)测试方法一致)，需要在导出模型时指定`-o exclude_nms=True`。测速需设置`--run_benchmark=True`, 首先需要安装以下依赖`pip install pynvml psutil GPUtil`。

**使用Paddle Inference但不使用TensorRT**进行测速，执行以下命令：

```bash
# 导出模型
python tools/export_model.py -c configs/yolov8/yolov8_s_500e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/yolov8_s_500e_coco.pdparams exclude_nms=True

# 速度测试，使用run_benchmark=True
python deploy/python/infer.py --model_dir=output_inference/yolov8_s_500e_coco --image_file=demo/000000014439_640x640.jpg --run_mode=paddle --device=gpu --run_benchmark=True
```

**使用Paddle Inference且使用TensorRT**进行测速，执行以下命令：

```bash
# 导出模型，使用trt=True
python tools/export_model.py -c configs/yolov8/yolov8_s_500e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/yolov8_s_500e_coco.pdparams exclude_nms=True trt=True

# 速度测试，使用run_benchmark=True
python deploy/python/infer.py --model_dir=output_inference/yolov8_s_500e_coco --image_file=demo/000000014439_640x640.jpg --device=gpu --run_benchmark=True

# tensorRT-FP32测速
python deploy/python/infer.py --model_dir=output_inference/yolov8_s_500e_coco --image_file=demo/000000014439_640x640.jpg --device=gpu --run_benchmark=True --run_mode=trt_fp32

# tensorRT-FP16测速
python deploy/python/infer.py --model_dir=output_inference/yolov8_s_500e_coco --image_file=demo/000000014439_640x640.jpg --device=gpu --run_benchmark=True --run_mode=trt_fp16
```
**注意:**
- 导出模型时指定`-o exclude_nms=True`仅作为测速时用，这样导出的模型其推理部署预测的结果不是最终检出框的结果。
- [模型库](#模型库)中的速度测试结果为tensorRT-FP16测速后的最快速度，为不包含数据预处理和模型输出后处理(NMS)的耗时。


## 引用
```

```
