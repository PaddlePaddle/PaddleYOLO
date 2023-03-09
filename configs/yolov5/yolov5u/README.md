# YOLOv5u

### YOLOv5u 模型

| 网络网络        | 输入尺寸   | 图片数/GPU | 学习率策略 | 模型推理耗时(ms) | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | Params(M) | FLOPs(G) |    下载链接       | 配置文件 |
| :------------- | :------- | :-------: | :------: | :------------: | :---------------------: | :----------------: |:---------: | :------: |:---------------: |:-----: |
| YOLOv5u-n        |  640     |    16      |   300e   |     1.61    |  34.5  | 49.7 |  2.65  | 7.79 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov5u_n_300e_coco.pdparams) | [配置文件](./yolov5u_n_300e_coco.yml) |
| YOLOv5u-s        |  640     |    16      |   300e   |     2.66    |  43.0  | 59.7 |  9.15   | 24.12 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov5u_s_300e_coco.pdparams) | [配置文件](./yolov5u_s_300e_coco.yml) |
| YOLOv5u-m        |  640     |    16      |   300e   |     5.50    |  49.0  | 65.7 |  25.11  | 64.42 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov5u_m_300e_coco.pdparams) | [配置文件](./yolov5u_m_300e_coco.yml) |
| YOLOv5u-l        |  640     |    16      |   300e   |     8.73    |  52.2  | 69.0 |  53.23  | 135.34 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov5u_l_300e_coco.pdparams) | [配置文件](./yolov5u_l_300e_coco.yml) |
| YOLOv5u-x        |  640     |    16      |   300e   |     15.49   |  53.1  | 69.9 |  97.28  | 246.89 | [下载链接](https://paddledet.bj.bcebos.com/models/yolov5u_x_300e_coco.pdparams) | [配置文件](./yolov5u_x_300e_coco.yml) |

**注意:**
  - YOLOv5u 模型表示YOLOv5结构使用YOLOv8的head和loss，是Anchor Free的检测方案，YOLOv5u 模型暂未支持完全训练；
  - YOLOv5u 模型训练使用COCO train2017作为训练集，Box AP为在COCO val2017上的`mAP(IoU=0.5:0.95)`结果；
