# RTMDet

## 内容
- [模型库](#模型库)
- [使用说明](#使用说明)
- [速度测试](#速度测试)
- [引用](#引用)

## 模型库
### RTMDet on COCO

| 网络网络        | 输入尺寸   | 图片数/GPU | 学习率策略 | 模型推理耗时(ms) |   mAP  |   AP50  | Params(M) | FLOPs(G) |  下载链接       | 配置文件 |
| :------------- | :------- | :-------: | :------: | :---------: | :-----: |:-----: | :-----: |:-----: | :-------------: | :-----: |
| *RTMDet-t       |  640     |    32      |   300e    |     -    |  40.9 | 57.9 |  4.90  | 16.21 |[下载链接](https://paddledet.bj.bcebos.com/models/rtmdet_t_300e_coco.pdparams) | [配置文件](./rtmdet_t_300e_coco.yml) |
| *RTMDet-s       |  640     |    32      |   300e    |     -    |  44.5 | 62.0 |  8.89  | 29.71 |[下载链接](https://paddledet.bj.bcebos.com/models/rtmdet_s_300e_coco.pdparams) | [配置文件](./rtmdet_s_300e_coco.yml) |
| *RTMDet-m       |  640     |    32      |   300e    |     -    |  49.1 | 66.8 |  24.71  | 78.47 |[下载链接](https://paddledet.bj.bcebos.com/models/rtmdet_m_300e_coco.pdparams) | [配置文件](./rtmdet_m_300e_coco.yml) |
| *RTMDet-l       |  640     |    32      |   300e    |     -    |  51.2 | 68.8 |  52.31  | 160.32 |[下载链接](https://paddledet.bj.bcebos.com/models/rtmdet_l_300e_coco.pdparams) | [配置文件](./rtmdet_l_300e_coco.yml) |
| *RTMDet-x       |  640     |    32      |   300e    |     -    |  52.6 | 70.4 |  94.86  | 283.12 |[下载链接](https://paddledet.bj.bcebos.com/models/rtmdet_x_300e_coco.pdparams) | [配置文件](./rtmdet_x_300e_coco.yml) |


### 部署模型

| 网络模型     | 输入尺寸 | 导出后的权重(w/o NMS) | ONNX(w/o NMS)  |
| :-------- | :--------: | :---------------------: | :----------------: |
| RTMDet-t |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_t_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_t_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_t_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_t_300e_coco_wo_nms.onnx) |
| RTMDet-s |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_s_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_s_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_s_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_s_300e_coco_wo_nms.onnx) |
| RTMDet-m |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_m_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_m_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_m_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_m_300e_coco_wo_nms.onnx) |
| RTMDet-l |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_l_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_l_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_l_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_l_300e_coco_wo_nms.onnx) |
| RTMDet-x |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_x_300e_coco_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_x_300e_coco_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_x_300e_coco_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/yoloseries/rtmdet/rtmdet_x_300e_coco_wo_nms.onnx) |


## 使用教程

### **一键运行全流程**:
```
model_type=rtmdet # 可修改，如 yolov7
job_name=rtmdet_s_300e_coco # 可修改，如 rtmdet_l_300e_coco

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

### 1. 训练
执行以下指令使用混合精度训练rtmdet
```bash
python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/rtmdet/rtmdet_s_300e_coco.yml --amp --eval
```
**注意:**
- `--amp`表示开启混合精度训练以避免显存溢出，`--eval`表示边训边验证。

### 2. 评估
执行以下命令在单个GPU上评估COCO val2017数据集
```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/rtmdet/rtmdet_s_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/rtmdet_s_300e_coco.pdparams
```

### 3. 推理
使用以下命令在单张GPU上预测图片，使用`--infer_img`推理单张图片以及使用`--infer_dir`推理文件中的所有图片。
```bash
# 推理单张图片
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/rtmdet/rtmdet_s_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/rtmdet_s_300e_coco.pdparams --infer_img=demo/000000014439_640x640.jpg

# 推理文件中的所有图片
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/rtmdet/rtmdet_s_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/rtmdet_s_300e_coco.pdparams --infer_dir=demo
```

### 4.导出模型
在GPU上推理部署或benchmark测速等需要通过`tools/export_model.py`导出模型。

当你**使用Paddle Inference但不使用TensorRT**时，运行以下的命令导出模型

```bash
python tools/export_model.py -c configs/rtmdet/rtmdet_s_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/rtmdet_s_300e_coco.pdparams
```

当你**使用Paddle Inference且使用TensorRT**时，需要指定`-o trt=True`来导出模型。

```bash
python tools/export_model.py -c configs/rtmdet/rtmdet_s_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/rtmdet_s_300e_coco.pdparams trt=True
```

如果你想将rtmdet模型导出为**ONNX格式**，参考
[PaddleDetection模型导出为ONNX格式教程](../../deploy/EXPORT_ONNX_MODEL.md)，运行以下命令：

```bash

# 导出推理模型
python tools/export_model.py -c configs/rtmdet/rtmdet_s_300e_coco.yml --output_dir=output_inference -o weights=https://paddledet.bj.bcebos.com/models/rtmdet_s_300e_coco.pdparams

# 安装paddle2onnx
pip install paddle2onnx

# 转换成onnx格式
paddle2onnx --model_dir output_inference/rtmdet_s_300e_coco --model_filename model.pdmodel --params_filename model.pdiparams --opset_version 11 --save_file rtmdet_s_300e_coco.onnx
```

**注意：** ONNX模型目前只支持batch_size=1


### 5.推理部署
rtmdet可以使用以下方式进行部署：
  - Paddle Inference [Python](../../deploy/python) & [C++](../../deploy/cpp)
  - [Paddle-TensorRT](../../deploy/TENSOR_RT.md)
  - [PaddleServing](https://github.com/PaddlePaddle/Serving)
  - [PaddleSlim模型量化](../slim)

运行以下命令导出模型

```bash
python tools/export_model.py -c configs/rtmdet/rtmdet_s_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/rtmdet_s_300e_coco.pdparams trt=True
```

**注意：**
- trt=True表示**使用Paddle Inference且使用TensorRT**进行测速，速度会更快，默认不加即为False，表示**使用Paddle Inference但不使用TensorRT**进行测速。
- 如果是使用Paddle Inference在TensorRT FP16模式下部署，需要参考[Paddle Inference文档](https://www.paddlepaddle.org.cn/inference/master/user_guides/download_lib.html#python)，下载并安装与你的CUDA, CUDNN和TensorRT相应的wheel包。

#### 5.1.Python部署
`deploy/python/infer.py`使用上述导出后的Paddle Inference模型用于推理和benchnark测速，如果设置了`--run_benchmark=True`, 首先需要安装以下依赖`pip install pynvml psutil GPUtil`。

```bash
# Python部署推理单张图片
python deploy/python/infer.py --model_dir=output_inference/rtmdet_s_300e_coco --image_file=demo/000000014439_640x640.jpg --device=gpu

# 推理文件夹下的所有图片
python deploy/python/infer.py --model_dir=output_inference/rtmdet_s_300e_coco --image_dir=demo/ --device=gpu
```

#### 5.2. C++部署
`deploy/cpp/build/main`使用上述导出后的Paddle Inference模型用于C++推理部署, 首先按照[docs](../../deploy/cpp/docs)编译安装环境。
```bash
# C++部署推理单张图片
./deploy/cpp/build/main --model_dir=output_inference/rtmdet_s_300e_coco/ --image_file=demo/000000014439_640x640.jpg --run_mode=paddle --device=GPU --threshold=0.5 --output_dir=cpp_infer_output/rtmdet_s_300e_coco
```


## 速度测试

为了公平起见，在[模型库](#模型库)中的速度测试结果均为不包含数据预处理和模型输出后处理(NMS)的数据(与[YOLOv4(AlexyAB)](https://github.com/AlexeyAB/darknet)测试方法一致)，需要在导出模型时指定`-o exclude_nms=True`。测速需设置`--run_benchmark=True`, 首先需要安装以下依赖`pip install pynvml psutil GPUtil`。

**使用Paddle Inference但不使用TensorRT**进行测速，执行以下命令：

```bash
# 导出模型
python tools/export_model.py -c configs/rtmdet/rtmdet_s_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/rtmdet_s_300e_coco.pdparams exclude_nms=True

# 速度测试，使用run_benchmark=True
python deploy/python/infer.py --model_dir=output_inference/rtmdet_s_300e_coco --image_file=demo/000000014439_640x640.jpg --run_mode=paddle --device=gpu --run_benchmark=True
```

**使用Paddle Inference且使用TensorRT**进行测速，执行以下命令：

```bash
# 导出模型，使用trt=True
python tools/export_model.py -c configs/rtmdet/rtmdet_s_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/rtmdet_s_300e_coco.pdparams exclude_nms=True trt=True

# 速度测试，使用run_benchmark=True
python deploy/python/infer.py --model_dir=output_inference/rtmdet_s_300e_coco --image_file=demo/000000014439_640x640.jpg --device=gpu --run_benchmark=True

# tensorRT-FP32测速
python deploy/python/infer.py --model_dir=output_inference/rtmdet_s_300e_coco --image_file=demo/000000014439_640x640.jpg --device=gpu --run_benchmark=True --run_mode=trt_fp32

# tensorRT-FP16测速
python deploy/python/infer.py --model_dir=output_inference/rtmdet_s_300e_coco --image_file=demo/000000014439_640x640.jpg --device=gpu --run_benchmark=True  --run_mode=trt_fp16
```
**注意:**
- 导出模型时指定`-o exclude_nms=True`仅作为测速时用，这样导出的模型其推理部署预测的结果不是最终检出框的结果。
- [模型库](#模型库)中的速度测试结果为tensorRT-FP16测速后的最快速度，为不包含数据预处理和模型输出后处理(NMS)的耗时。


## 引用
```

```
