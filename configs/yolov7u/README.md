# YOLOv7u

#### YOLOv7u 模型

| 网络网络        | 输入尺寸   | 图片数/GPU | 学习率策略 | 模型推理耗时(ms) | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | Params(M) | FLOPs(G) |    下载链接       | 配置文件 |
| :------------- | :------- | :-------: | :------: | :------------: | :---------------------: | :----------------: |:---------: | :------: |:---------------: |:-----: |
| YOLOv7u-L     |  640     |    16      |   300e    |       9.0      |  52.1 | 68.8 |  43.59  | 130.10 |[下载链接](https://paddledet.bj.bcebos.com/models/yolov7u_l_300e_coco.pdparams) | [配置文件](./yolov7u_l_300e_coco.yml) |


**注意:**
  - YOLOv7u 模型表示YOLOv7结构使用YOLOv8的head和loss，并结合YOLOR的ImplicitA和ImplicitM，是Anchor Free的检测方案；
  - YOLOv7u 模型训练使用COCO train2017作为训练集，Box AP为在COCO val2017上的`mAP(IoU=0.5:0.95)`结果；
  - 使用教程可参照[YOLOv7](../yolov7)；
