# YOLOv8

## 内容
- [模型库](#模型库)
- [使用教程](#使用教程)
- [FastDeploy多硬件快速部署](#FastDeploy多硬件快速部署)
- [引用](#引用)


## 模型库

### 基础模型

| 网络网络        | 输入尺寸   | 图片数/GPU | 学习率策略 | TRT-FP16-Latency(ms) | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | Params(M) | FLOPs(G) |    下载链接       | 配置文件 |
| :------------- | :------- | :-------: | :------: | :------------: | :---------------------: | :----------------: |:---------: | :------: |:---------------: |:-----: |
| *YOLOv8-n        |  640     |    16      |   500e   |    1.8   |  37.3  | 53.0 |  3.16   | 8.7 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov8_n_500e_coco.pdparams) | [配置文件](./yolov8_n_500e_coco.yml) |
| *YOLOv8-s        |  640     |    16      |   500e   |    3.4   |  44.9  | 61.8 |  11.17  | 28.6 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov8_s_500e_coco.pdparams) | [配置文件](./yolov8_s_500e_coco.yml) |
| *YOLOv8-m        |  640     |    16      |   500e   |    6.5   |  50.2  | 67.3 |  25.90  | 78.9 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov8_m_500e_coco.pdparams) | [配置文件](./yolov8_m_500e_coco.yml) |
| *YOLOv8-l        |  640     |    16      |   500e   |    10.0  |  52.8  | 69.6 |  43.69  | 165.2 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov8_l_500e_coco.pdparams) | [配置文件](./yolov8_l_500e_coco.yml) |
| *YOLOv8-x        |  640     |    16      |   500e   |    15.1  |  53.8  | 70.6 |  68.23  | 257.8 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov8_x_500e_coco.pdparams) | [配置文件](./yolov8_x_500e_coco.yml) |

### P6大尺度模型

| 网络网络        | 输入尺寸   | 图片数/GPU | 学习率策略 | TRT-FP16-Latency(ms) | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | Params(M) | FLOPs(G) |    下载链接       | 配置文件 |
| :------------- | :------- | :-------: | :------: | :------------: | :---------------------: | :----------------: |:---------: | :------: |:---------------: |:-----: |
| *YOLOv8-P6-x     |  1280    |    16      |   500e   |    55.0  |  -  | - |  97.42  | 522.93 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov8p6_x_500e_coco.pdparams) | [配置文件](./yolov8p6_x_500e_coco.yml) |


**注意:**
  - YOLOv8模型暂未支持完全训练，mAP为部署权重在COCO val2017上的`mAP(IoU=0.5:0.95)`结果，且评估未使用`multi_label`等trick；
  - YOLOv8模型训练使用COCO train2017作为训练集，Box AP为在COCO val2017上的`mAP(IoU=0.5:0.95)`结果；
  - YOLOv8模型训练过程中默认使用8 GPUs进行混合精度训练，默认lr为0.01为8卡总batch_size的设置，如果**GPU卡数**或者每卡**batch size**发生改动，也不需要改动学习率，但为了保证高精度最好使用**总batch size大于64**的配置去训练；
  - TRT-FP16-Latency(ms)模型推理耗时为TensorRT-FP16下测试的耗时，不包含数据预处理和模型输出后处理(NMS)的耗时。测试采用**单卡Tesla T4 GPU**，batch size=1，测试环境为**paddlepaddle-2.3.2**, **CUDA 11.2**, **CUDNN 8.2**, **GCC-8.2**, **TensorRT 8.0.3.4**。
  - 如果你设置了`--run_benchmark=True`, 你首先需要安装以下依赖`pip install pynvml psutil GPUtil`。

### 部署模型

| 网络模型   | 输入尺寸 | 导出后的权重(带nms) | 导出后的权重(exclude_nms)| ONNX(exclude_post_process)  |
| :-------- | :----: | :---------------: | :--------------------: | :-------------------------: |
| YOLOv8-n |  640   | [(w_nms)](https://paddledet.bj.bcebos.com/deploy/paddleyolo/yolov8/yolov8_n_500e_coco_w_nms.zip) | [(wo_nms)](https://paddledet.bj.bcebos.com/deploy/paddleyolo/yolov8/yolov8_n_500e_coco_wo_nms.zip) | [(onnx)](https://paddledet.bj.bcebos.com/deploy/paddleyolo/yolov8/yolov8_n_500e_coco.onnx) |
| YOLOv8-s |  640   | [(w_nms)](https://paddledet.bj.bcebos.com/deploy/paddleyolo/yolov8/yolov8_s_500e_coco_w_nms.zip) | [(wo_nms)](https://paddledet.bj.bcebos.com/deploy/paddleyolo/yolov8/yolov8_s_500e_coco_wo_nms.zip) | [(onnx)](https://paddledet.bj.bcebos.com/deploy/paddleyolo/yolov8/yolov8_s_500e_coco.onnx) |
| YOLOv8-m |  640   | [(w_nms)](https://paddledet.bj.bcebos.com/deploy/paddleyolo/yolov8/yolov8_m_500e_coco_w_nms.zip) | [(wo_nms)](https://paddledet.bj.bcebos.com/deploy/paddleyolo/yolov8/yolov8_m_500e_coco_wo_nms.zip) | [(onnx)](https://paddledet.bj.bcebos.com/deploy/paddleyolo/yolov8/yolov8_m_500e_coco.onnx) |
| YOLOv8-l |  640   | [(w_nms)](https://paddledet.bj.bcebos.com/deploy/paddleyolo/yolov8/yolov8_l_500e_coco_w_nms.zip) | [(wo_nms)](https://paddledet.bj.bcebos.com/deploy/paddleyolo/yolov8/yolov8_l_500e_coco_wo_nms.zip) | [(onnx)](https://paddledet.bj.bcebos.com/deploy/paddleyolo/yolov8/yolov8_l_500e_coco.onnx) |
| YOLOv8-x |  640   | [(w_nms)](https://paddledet.bj.bcebos.com/deploy/paddleyolo/yolov8/yolov8_x_500e_coco_w_nms.zip) | [(wo_nms)](https://paddledet.bj.bcebos.com/deploy/paddleyolo/yolov8/yolov8_x_500e_coco_wo_nms.zip) | [(onnx)](https://paddledet.bj.bcebos.com/deploy/paddleyolo/yolov8/yolov8_x_500e_coco.onnx) |

**注意:**
 - 带nms的导出权重为普通导出方式，加trt表示用于trt加速，对NMS和silu激活函数提速明显。运行命令为：
  ```CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c ${config} -o weights=${weights} trt=True```
 - `exclude_nms`导出的权重表示去除NMS导出，返回2个Tensor，是缩放回原图后的坐标和分类置信度。运行命令为：
  ```CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c ${config} -o weights=${weights} exclude_nms=True trt=True```
  - `exclude_post_process`导出表示去除后处理导出，返回和YOLOv5导出ONNX时相同格式的concat后的1个Tensor，是未缩放回原图的坐标和分类置信度。运行命令为：
  ```CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c ${config} -o weights=${weights} exclude_post_process=True trt=True ```


## 使用教程

### 0. **一键运行全流程**

将以下命令写在一个脚本文件里如```run.sh```，一键运行命令为：```sh run.sh```，也可命令行一句句去运行。

```bash
model_name=yolov8 # 可修改，如 ppyoloe
job_name=yolov8_s_500e_coco # 可修改，如 ppyoloe_plus_crn_s_80e_coco

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


## FastDeploy多硬件快速部署

FastDeploy是飞桨推出的统一部署工具，支持云边端部署。目前在YOLO系列支持的部署能力如下所示。具体部署示例，可以前往[FastDeploy仓库](https://github.com/PaddlePaddle/FastDeploy)使用。

|                                                                                                                                | [YOLOv5](https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/vision/detection/paddledetection) | [YOLOv6](https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/vision/detection/paddledetection) | [YOLOv7](https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/vision/detection/paddledetection) | [YOLOv8](https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/vision/detection/paddledetection) | [PP-YOLOE+](https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/vision/detection/paddledetection) | 部署特色                       |
| ------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| [Intel CPU](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/download_prebuilt_libraries.md)  | 支持                                                                                                          | 支持                                                                                                          | 支持                                                                                                          | 支持                                                                                                          | 支持                                                                                                             | 集成PaddleSlim一键压缩压缩，实现极致性能             |
| [NVIDIA GPU](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/download_prebuilt_libraries.md) | 支持                                                                                                          | 支持                                                                                                          | 支持                                                                                                          | 支持                                                                                                          | 支持                                                                                                             | 集成PaddleSlim一键压缩工具、CUDA预处理加速，实现极致性能   |
| [飞腾 CPU]()                                                                                                                     | 支持                                                                                                          | 支持                                                                                                          | 支持                                                                                                          | 支持                                                                                                          | 支持                                                                                                             | X86 CPU与ARM CPU无缝切换                   |
| [昆仑芯 R200*](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/kunlunxin.md)                    | 支持                                                                                                          | 支持                                                                                                          | 支持                                                                                                          | 支持                                                                                                          | 支持                                                                                                             | 无缝部署Paddle模型                          |
| [昇腾310*](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/kunlunxin.md)                       | 支持                                                                                                          | 即将支持                                                                                                        | 即将支持                                                                                                        | 即将支持                                                                                                        | 支持                                                                                                             | 无缝部署Paddle模型                          |
| [算能SC7-FP300*](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/sophgo.md)                    | 支持                                                                                                          | 支持                                                                                                          | 支持                                                                                                          | 支持                                                                                                          | 支持                                                                                                             | 充分发挥硬件工具链特性，实现模型快速部署                  |
| [Jetson](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/jetson.md)                          | 支持                                                                                                          | 支持                                                                                                          | 支持                                                                                                          | 支持                                                                                                          | 支持                                                                                                             | 集成PaddleSlim一键压缩工具、CUDA预处理加速，实现极致性能   |
| [ARM CPU](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/download_prebuilt_libraries.md)    | 支持                                                                                                          | 支持                                                                                                          | 支持                                                                                                          | 支持                                                                                                          | 支持                                                                                                             | 集成PaddleSlim一键压缩工具、预处理加速库FlyCV，实现极致性能 |
| [RK3588*](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/rknpu2.md)                         | 支持                                                                                                          | 支持                                                                                                          | 支持                                                                                                          | 支持                                                                                                          | 支持                                                                                                             | 充分发挥硬件工具链特性，实现模型快速部署                  |
| [RV1126*](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/rv1126.md)                         | 支持                                                                                                          | 暂不支持                                                                                                        | 暂不支持                                                                                                        | 暂不支持                                                                                                        | 支持                                                                                                             | 联合全量化实现模型端到端的优化                       |
| [服务化部署](https://github.com/PaddlePaddle/FastDeploy/tree/develop/serving)                                                       | 支持                                                                                                          | 暂不支持                                                                                                        | 暂不支持                                                                                                        | 暂不支持                                                                                                        | 支持                                                                                                             | 实现企业级高并发需求                            |
| [视频流部署](https://github.com/PaddlePaddle/FastDeploy/tree/develop/streamer)                                                      | 暂不支持                                                                                                        | 暂不支持                                                                                                        | 暂不支持                                                                                                        | 暂不支持                                                                                                        | 支持                                                                                                             | 调用硬解码核，实现数据零拷贝，充分利用硬件资源               |

备注：

*表示：FastDeploy目前在该型号硬件上测试。通常同类型硬件上使用的是相同的软件栈，该部署能力可以延伸到同软件架栈的硬件。譬如RK3588与RK3566、RK3568相同的软件栈。

「硬件列-纵轴」链接到部署预编译包安装或部署示例，「横轴」跳转到具体部署示例。


## 引用
```

```
