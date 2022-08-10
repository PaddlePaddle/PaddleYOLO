# MT-YOLOv6

## 内容
- [模型库](#模型库)
- [使用说明](#使用说明)
- [速度测试](#速度测试)

## 模型库
### MT-YOLOv6 on COCO

| 网络网络        | 输入尺寸   | 图片数/GPU | 学习率策略 | 模型推理耗时(ms) |   mAP  |   AP50  | Params(M) | FLOPs(G) |  下载链接       | 配置文件 |
| :------------- | :------- | :-------: | :------: | :---------: | :-----: |:-----: | :-----: |:-----: | :-------------: | :-----: |
| YOLOv6mt-n       |  416     |    32      |   400e    |     2.5    | 30.5  |    46.8 |  4.74  | 5.16 |[下载链接](https://paddledet.bj.bcebos.com/models/yolov6mt_n_416_400e_coco.pdparams) | [配置文件](./yolov6mt_n_416_400e_coco.yml) |
| YOLOv6mt-n       |  640     |    32      |   400e    |     2.8    |  34.7 |    52.7 |  4.74  | 12.2 |[下载链接](https://paddledet.bj.bcebos.com/models/yolov6mt_n_400e_coco.pdparams) | [配置文件](./yolov6mt_n_400e_coco.yml) |
| YOLOv6mt-t       |  640     |    32      |   400e    |     2.9    |  40.8 |  60.4 |  16.36  | 39.94 |[下载链接](https://paddledet.bj.bcebos.com/models/yolov6mt_t_400e_coco.pdparams) | [配置文件](./yolov6mt_t_400e_coco.yml) |
| YOLOv6mt-s       |  640     |    32      |   400e    |     3.0    | 42.5 |    61.7 |  18.87  | 48.36 |[下载链接](https://paddledet.bj.bcebos.com/models/yolov6mt_s_400e_coco.pdparams) | [配置文件](./yolov6mt_s_400e_coco.yml) |


**注意:**
  - MT-YOLOv6模型训练使用COCO train2017作为训练集，Box AP为在COCO val2017上的`mAP(IoU=0.5:0.95)`结果；
  - MT-YOLOv6模型训练过程中默认使用8 GPUs进行混合精度训练，默认lr为0.01为8卡总batch_size的设置，如果**GPU卡数**或者每卡**batch size**发生改动，也不需要改动学习率，但为了保证高精度最好使用**总batch size大于64**的配置去训练；
  - 模型推理耗时(ms)为TensorRT-FP16下测试的耗时，不包含数据预处理和模型输出后处理(NMS)的耗时。测试采用单卡V100，batch size=1，测试环境为**paddlepaddle-2.3.0**, **CUDA 11.2**, **CUDNN 8.2**, **GCC-8.2**, **TensorRT 8.0.3.4**，具体请参考[速度测试](#速度测试)。
  - 如果你设置了`--run_benchmark=True`, 你首先需要安装以下依赖`pip install pynvml psutil GPUtil`。


## 使用教程

### 1. 训练
执行以下指令使用混合精度训练MT-YOLOv6
```bash
python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/yolov6mt/yolov6mt_s_400e_coco.yml --amp --eval
```
**注意:**
- `--amp`表示开启混合精度训练以避免显存溢出，`--eval`表示边训边验证。

### 2. 评估
执行以下命令在单个GPU上评估COCO val2017数据集
```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/yolov6mt/yolov6mt_s_400e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/yolov6mt_s_400e_coco.pdparams
```

### 3. 推理
使用以下命令在单张GPU上预测图片，使用`--infer_img`推理单张图片以及使用`--infer_dir`推理文件中的所有图片。
```bash
# 推理单张图片
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/yolov6mt/yolov6mt_s_400e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/yolov6mt_s_400e_coco.pdparams --infer_img=demo/000000014439_640x640.jpg

# 推理文件中的所有图片
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/yolov6mt/yolov6mt_s_400e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/yolov6mt_s_400e_coco.pdparams --infer_dir=demo
```

### 4.导出模型
在GPU上推理部署或benchmark测速等需要通过`tools/export_model.py`导出模型。

当你**使用Paddle Inference但不使用TensorRT**时，运行以下的命令导出模型

```bash
python tools/export_model.py -c configs/yolov6mt/yolov6mt_s_400e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/yolov6mt_s_400e_coco.pdparams
```

当你**使用Paddle Inference且使用TensorRT**时，需要指定`-o trt=True`来导出模型。

```bash
python tools/export_model.py -c configs/yolov6mt/yolov6mt_s_400e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/yolov6mt_s_400e_coco.pdparams trt=True
```

如果你想将MT-YOLOv6模型导出为**ONNX格式**，参考
[PaddleDetection模型导出为ONNX格式教程](../../deploy/EXPORT_ONNX_MODEL.md)，运行以下命令：

```bash

# 导出推理模型
python tools/export_model.py -c configs/yolov6mt/yolov6mt_s_400e_coco.yml --output_dir=output_inference -o weights=https://paddledet.bj.bcebos.com/models/yolov6mt_s_400e_coco.pdparams

# 安装paddle2onnx
pip install paddle2onnx

# 转换成onnx格式
paddle2onnx --model_dir output_inference/yolov6mt_s_400e_coco --model_filename model.pdmodel --params_filename model.pdiparams --opset_version 11 --save_file yolov6mt_s_400e_coco.onnx
```

**注意：** ONNX模型目前只支持batch_size=1


### 5.推理部署
MT-YOLOv6可以使用以下方式进行部署：
  - Paddle Inference [Python](../../deploy/python) & [C++](../../deploy/cpp)
  - [Paddle-TensorRT](../../deploy/TENSOR_RT.md)
  - [PaddleServing](https://github.com/PaddlePaddle/Serving)
  - [PaddleSlim模型量化](../slim)

运行以下命令导出模型

```bash
python tools/export_model.py -c configs/yolov6mt/yolov6mt_s_400e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/yolov6mt_s_400e_coco.pdparams trt=True
```

**注意：**
- trt=True表示**使用Paddle Inference且使用TensorRT**进行测速，速度会更快，默认不加即为False，表示**使用Paddle Inference但不使用TensorRT**进行测速。
- 如果是使用Paddle Inference在TensorRT FP16模式下部署，需要参考[Paddle Inference文档](https://www.paddlepaddle.org.cn/inference/master/user_guides/download_lib.html#python)，下载并安装与你的CUDA, CUDNN和TensorRT相应的wheel包。

#### 5.1.Python部署
`deploy/python/infer.py`使用上述导出后的Paddle Inference模型用于推理和benchnark测速，如果设置了`--run_benchmark=True`, 首先需要安装以下依赖`pip install pynvml psutil GPUtil`。

```bash
# Python部署推理单张图片
python deploy/python/infer.py --model_dir=output_inference/yolov6mt_s_400e_coco --image_file=demo/000000014439_640x640.jpg --device=gpu

# 推理文件夹下的所有图片
python deploy/python/infer.py --model_dir=output_inference/yolov6mt_s_400e_coco --image_dir=demo/ --device=gpu
```

#### 5.2. C++部署
`deploy/cpp/build/main`使用上述导出后的Paddle Inference模型用于C++推理部署, 首先按照[docs](../../deploy/cpp/docs)编译安装环境。
```bash
# C++部署推理单张图片
./deploy/cpp/build/main --model_dir=output_inference/yolov6mt_s_400e_coco/ --image_file=demo/000000014439_640x640.jpg --run_mode=paddle --device=GPU --threshold=0.5 --output_dir=cpp_infer_output/yolov6mt_s_400e_coco
```


## 速度测试

为了公平起见，在[模型库](#模型库)中的速度测试结果均为不包含数据预处理和模型输出后处理(NMS)的数据(与[YOLOv4(AlexyAB)](https://github.com/AlexeyAB/darknet)测试方法一致)，需要在导出模型时指定`-o exclude_nms=True`。测速需设置`--run_benchmark=True`, 首先需要安装以下依赖`pip install pynvml psutil GPUtil`。

**使用Paddle Inference但不使用TensorRT**进行测速，执行以下命令：

```bash
# 导出模型
python tools/export_model.py -c configs/yolov6mt/yolov6mt_s_400e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/yolov6mt_s_400e_coco.pdparams exclude_nms=True

# 速度测试，使用run_benchmark=True
python deploy/python/infer.py --model_dir=output_inference/yolov6mt_s_400e_coco --image_file=demo/000000014439_640x640.jpg --run_mode=paddle --device=gpu --run_benchmark=True
```

**使用Paddle Inference且使用TensorRT**进行测速，执行以下命令：

```bash
# 导出模型，使用trt=True
python tools/export_model.py -c configs/yolov6mt/yolov6mt_s_400e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/yolov6mt_s_400e_coco.pdparams exclude_nms=True trt=True

# 速度测试，使用run_benchmark=True
python deploy/python/infer.py --model_dir=output_inference/yolov6mt_s_400e_coco --image_file=demo/000000014439_640x640.jpg --device=gpu --run_benchmark=True

# tensorRT-FP32测速
python deploy/python/infer.py --model_dir=output_inference/yolov6mt_s_400e_coco --image_file=demo/000000014439_640x640.jpg --device=gpu --run_benchmark=True --run_mode=trt_fp32

# tensorRT-FP16测速
python deploy/python/infer.py --model_dir=output_inference/yolov6mt_s_400e_coco --image_file=demo/000000014439_640x640.jpg --device=gpu --run_benchmark=True  --run_mode=trt_fp16
```
**注意:**
- 导出模型时指定`-o exclude_nms=True`仅作为测速时用，这样导出的模型其推理部署预测的结果不是最终检出框的结果。
- [模型库](#模型库)中的速度测试结果为tensorRT-FP16测速后的最快速度，为不包含数据预处理和模型输出后处理(NMS)的耗时。
