# YOLOv8 Instance segmentation

## 模型库

### 实例分割模型

| 网络网络        | 输入尺寸   | 图片数/GPU | 学习率策略 | TRT-FP16-Latency(ms) | box AP | mask AP | Params(M) | FLOPs(G) |    下载链接       | 配置文件 |
| :------------- | :------- | :-------: | :------: | :------------: | :---------------------: | :----------------: |:---------: | :------: |:---------------: |:-----: |
| *YOLOv8-n        |  640     |    16      |   500e   |    -   |  36.6  | - |  3.4   | 12.6 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov8_seg_n_500e_coco.pdparams) | [配置文件](./yolov8_seg_n_500e_coco.yml) |
| *YOLOv8-s        |  640     |    16      |   500e   |    -   |  44.6  | - |  11.8  | 42.6 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov8_seg_s_500e_coco.pdparams) | [配置文件](./yolov8_seg_s_500e_coco.yml) |
| *YOLOv8-m        |  640     |    16      |   500e   |    -   |  49.7  | - |  27.3  | 110.2 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov8_seg_m_500e_coco.pdparams) | [配置文件](./yolov8_seg_m_500e_coco.yml) |
| *YOLOv8-l        |  640     |    16      |   500e   |    -   |  52.1  | - |  46.0  | 220.5 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov8_seg_l_500e_coco.pdparams) | [配置文件](./yolov8_seg_l_500e_coco.yml) |
| *YOLOv8-x        |  640     |    16      |   500e   |    -   |  53.4  | - |  71.8  | 344.1 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov8_seg_x_500e_coco.pdparams) | [配置文件](./yolov8_seg_x_500e_coco.yml) |


**注意:**
  - YOLOv8模型mAP为部署权重在COCO val2017上的`mAP(IoU=0.5:0.95)`结果，且评估未使用`multi_label`等trick；
  - YOLOv8模型训练使用COCO train2017作为训练集，Box AP为在COCO val2017上的`mAP(IoU=0.5:0.95)`结果；
  - YOLOv8模型训练过程中默认使用8 GPUs进行混合精度训练，默认lr为0.01为8卡总batch_size的设置，如果**GPU卡数**或者每卡**batch size**发生改动，也不需要改动学习率，但为了保证高精度最好使用**总batch size大于64**的配置去训练；
  - TRT-FP16-Latency(ms)模型推理耗时为TensorRT-FP16下测试的耗时，不包含数据预处理和模型输出后处理(NMS)的耗时。测试采用**单卡Tesla T4 GPU**，batch size=1，测试环境为**paddlepaddle-2.3.2**, **CUDA 11.2**, **CUDNN 8.2**, **GCC-8.2**, **TensorRT 8.0.3.4**。
