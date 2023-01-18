# YOLO on VOC

## 模型库

| 网络模型        | 输入尺寸   | 图片数/GPU | 学习率策略 | TRT-FP16-Latency(ms) | mAP(0.50,11point) | Params(M) | FLOPs(G) |    下载链接       | 配置文件 |
| :-----------: | :-------: | :-------: | :------: | :------------: | :---------------: | :------------------: |:-----------------: | :------: | :------: |
| YOLOv5-s        |  640     |    16     |   60e    |     3.2   |  80.3 |  7.24  | 16.54 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov5_s_60e_voc.pdparams) | [配置文件](./yolov5_s_60e_voc.yml) |
| YOLOv7-tiny     |  640     |    32     |   60e    |     2.6   |  80.2 |  6.23  | 6.90 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov7_tiny_60e_voc.pdparams) | [配置文件](./yolov7_tiny_60e_voc.yml) |
| YOLOX-s         |  640     |    8      |   40e    |     3.0   |  82.9 |  9.0   |  26.8 | [下载链接](https://paddledet.bj.bcebos.com/models/yolox_s_40e_voc.pdparams) | [配置文件](./yolox_s_40e_voc.yml) |
| PP-YOLOE+_s     |  640     |    8      |   30e    |     2.9   |  86.7 |  7.93  |  17.36 | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_s_30e_voc.pdparams) | [配置文件](./ppyoloe_plus_crn_s_30e_voc.yml) |


**注意:**
  - 所有YOLO模型均使用VOC数据集训练，mAP为`mAP(IoU=0.5)`的结果，且评估未使用`multi_label`等trick；
  - 所有YOLO模型均加载各自模型的COCO权重作为预训练，各个配置文件的配置均为默认使用8卡GPU，可作为自定义数据集设置参考，具体精度会因数据集而异；
  - YOLO检测模型建议**总`batch_size`至少大于`64`**去训练，如果资源不够请**换小模型**或**减小模型的输入尺度**，为了保障较高检测精度，**尽量不要尝试单卡训和总`batch_size`小于`64`训**；
  - Params(M)和FLOPs(G)均为训练时所测，YOLOv7没有s模型，故选用tiny模型；
  - TRT-FP16-Latency(ms)测速相关请查看各YOLO模型的config的主页；


## 使用教程

### 下载数据集：

下载PaddleDetection团队整理的VOC数据，并放置于`PaddleDetection/dataset/voc`
```
wget https://bj.bcebos.com/v1/paddledet/data/voc.zip
```

### 训练评估预测：

```
model_name=voc
job_name=ppyoloe_plus_crn_s_30e_voc # 可修改，如 yolov7_tiny_60e_voc

config=configs/${model_name}/${job_name}.yml
log_dir=log_dir/${job_name}
# weights=https://bj.bcebos.com/v1/paddledet/models/${job_name}.pdparams
weights=output/${job_name}/model_final.pdparams

# 1.训练（单卡/多卡）
# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c ${config} --eval --amp
python -m paddle.distributed.launch --log_dir=${log_dir} --gpus 0,1,2,3,4,5,6,7 tools/train.py -c ${config} --eval --amp

# 2.评估
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c ${config} -o weights=${weights} --classwise

# 3.预测
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c ${config} -o weights=${weights} --infer_img=demo/000000014439_640x640.jpg --draw_threshold=0.5
```
