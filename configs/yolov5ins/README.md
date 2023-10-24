# YOLOv5 Instance segmentation

## 模型库

### 实例分割模型

| 网络网络        | 输入尺寸   | 图片数/GPU | 学习率策略 | 模型推理耗时(ms) | box AP | mask AP | Params(M) | FLOPs(G) |    下载链接       | 配置文件 |
| :------------- | :------- | :-------: | :------: | :------------: | :---------------------: | :----------------: |:---------: | :------: |:---------------: |:-----: |
| YOLOv5-n        |  640     |    16      |   300e   |     -    |  27.6  | - |  2.0  | 7.1 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov5_ins_n_300e_coco.pdparams) | [配置文件](./yolov5_ins_n_300e_coco.yml) |
| YOLOv5-s        |  640     |    16      |   300e   |     -    |  37.6  | - |  7.8  | 26.4 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov5_ins_s_300e_coco.pdparams) | [配置文件](./yolov5_ins_s_300e_coco.yml) |
| YOLOv5-m        |  640     |    16      |   300e   |     -    |  45.0  | - |  22.0  | 70.8 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov5_ins_m_300e_coco.pdparams) | [配置文件](./yolov5_ins_m_300e_coco.yml) |
| YOLOv5-l        |  640     |    16      |   300e   |     -    |  48.9  | - |  47.9  | 147.7 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov5_ins_l_300e_coco.pdparams) | [配置文件](./yolov5_ins_l_300e_coco.yml) |
| YOLOv5-x        |  640     |    16      |   300e   |     -    |  50.6  | - |  88.8  | 265.7 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov5_ins_x_300e_coco.pdparams) | [配置文件](./yolov5_ins_x_300e_coco.yml) |


**注意:**
  - YOLOv5模型训练使用COCO train2017作为训练集，Box AP为在COCO val2017上的`mAP(IoU=0.5:0.95)`结果；
  - YOLOv5u 模型表示YOLOv5结构使用YOLOv8的head和loss，是Anchor Free的检测方案，具体可参照[YOLOv5u](../yolov5u)；
  - YOLOv5模型训练过程中默认使用8 GPUs进行混合精度训练，默认lr为0.01为8卡总batch_size的设置，如果**GPU卡数**或者每卡**batch size**发生改动，也不需要改动学习率，但为了保证高精度最好使用**总batch size大于64**的配置去训练；
  - 模型推理耗时(ms)为TensorRT-FP16下测试的耗时，不包含数据预处理和模型输出后处理(NMS)的耗时。测试采用单卡Tesla T4 GPU，batch size=1，测试环境为**paddlepaddle-2.3.2**, **CUDA 11.2**, **CUDNN 8.2**, **GCC-8.2**, **TensorRT 8.0.3.4**，具体请参考[速度测试](#速度测试)。
