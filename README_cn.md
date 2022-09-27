## 简介

**PaddleYOLO**是基于[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)的YOLO系列模型库，**只包含YOLO系列模型的相关代码**，支持`YOLOv3`,`PP-YOLO`,`PP-YOLOv2`,`PP-YOLOE`,`PP-YOLOE+`,`YOLOX`,`YOLOv5`,`YOLOv6`,`YOLOv7`等模型，欢迎一起使用和建设！

## 更新日志
* 【2022/09/26】发布[`PaddleYOLO`](https://github.com/PaddlePaddle/PaddleYOLO)模型套件，请参照[ModelZoo](docs/MODEL_ZOO_cn.md)；
* 【2022/09/19】支持[`YOLOv6`](configs/yolov6)新版，包括n/t/s/m/l模型；
* 【2022/08/23】发布`YOLOSeries`代码库: 支持`YOLOv3`,`PP-YOLOE`,`PP-YOLOE+`,`YOLOX`,`YOLOv5`,`YOLOv6`,`YOLOv7`等YOLO模型，支持`ConvNeXt`骨干网络高精度版`PP-YOLOE`,`YOLOX`和`YOLOv5`等模型，支持PaddleSlim无损加速量化训练`PP-YOLOE`,`YOLOv5`,`YOLOv6`和`YOLOv7`等模型，详情可阅读[此文章](https://mp.weixin.qq.com/s/Hki01Zs2lQgvLSLWS0btrA)；


**注意:**
 - **PaddleYOLO**代码库协议为**GPL 3.0**，[YOLOv5](configs/yolov5),[YOLOv7](configs/yolov7)和[YOLOv6](configs/yolov6)这3类模型代码不合入[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)，其余YOLO模型推荐在[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)中使用，**会最先发布PP-YOLO系列特色检测模型的最新进展**；；
 - **PaddleYOLO**代码库**推荐使用paddlepaddle-2.3.2以上的版本**，请参考[官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)下载对应适合版本，**Windows平台请安装paddle develop版本**；


## <img src="https://user-images.githubusercontent.com/48054808/157793354-6e7f381a-0aa6-4bb7-845c-9acf2ecc05c3.png" width="20"/> 产品动态

- 🔥 **2022.9.26：PaddleYOLO发布[release/2.5版本](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5)**
  - 💡 模型套件：
    - 发布[PaddleYOLO](https://github.com/PaddlePaddle/PaddleYOLO)模型套件: 支持`YOLOv3`,`PP-YOLOE`,`PP-YOLOE+`,`YOLOX`,`YOLOv5`,`YOLOv6`,`YOLOv7`等YOLO模型，支持`ConvNeXt`骨干网络高精度版`PP-YOLOE`,`YOLOX`和`YOLOv5`等模型，支持PaddleSlim无损加速量化训练`PP-YOLOE`,`YOLOv5`,`YOLOv6`和`YOLOv7`等模型；

- 🔥 **2022.8.26：PaddleDetection发布[release/2.5版本](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5)**
  - 🗳 特色模型：
    - 发布[PP-YOLOE+](configs/ppyoloe)，最高精度提升2.4% mAP，达到54.9% mAP，模型训练收敛速度提升3.75倍，端到端预测速度最高提升2.3倍；多个下游任务泛化性提升
    - 发布[PicoDet-NPU](configs/picodet)模型，支持模型全量化部署；新增[PicoDet](configs/picodet)版面分析模型
    - 发布[PP-TinyPose升级版](./configs/keypoint/tiny_pose/)增强版，在健身、舞蹈等场景精度提升9.1% AP，支持侧身、卧躺、跳跃、高抬腿等非常规动作
  - 🔮 场景能力：
    - 发布行人分析工具[PP-Human v2](./deploy/pipeline)，新增打架、打电话、抽烟、闯入四大行为识别，底层算法性能升级，覆盖行人检测、跟踪、属性三类核心算法能力，提供保姆级全流程开发及模型优化策略，支持在线视频流输入
    - 首次发布[PP-Vehicle](./deploy/pipeline)，提供车牌识别、车辆属性分析（颜色、车型）、车流量统计以及违章检测四大功能，兼容图片、在线视频流、视频输入，提供完善的二次开发文档教程
  - 💡 前沿算法：
    - 全面覆盖的[YOLO家族](https://github.com/PaddlePaddle/PaddleYOLO)经典与最新模型: 包括YOLOv3，百度飞桨自研的实时高精度目标检测检测模型PP-YOLOE，以及前沿检测算法YOLOv4、YOLOv5、YOLOX，YOLOv6及YOLOv7
    - 新增基于[ViT](configs/vitdet)骨干网络高精度检测模型，COCO数据集精度达到55.7% mAP；新增[OC-SORT](configs/mot/ocsort)多目标跟踪模型；新增[ConvNeXt](configs/convnext)骨干网络
  - 📋 产业范例：新增[智能健身](https://aistudio.baidu.com/aistudio/projectdetail/4385813)、[打架识别](https://aistudio.baidu.com/aistudio/projectdetail/4086987?channelType=0&channel=0)、[来客分析](https://aistudio.baidu.com/aistudio/projectdetail/4230123?channelType=0&channel=0)、车辆结构化范例

- 2022.3.24：PaddleDetection发布[release/2.4版本](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4)
  - 发布高精度云边一体SOTA目标检测模型[PP-YOLOE](configs/ppyoloe)，提供s/m/l/x版本，l版本COCO test2017数据集精度51.6%，V100预测速度78.1 FPS，支持混合精度训练，训练较PP-YOLOv2加速33%，全系列多尺度模型，满足不同硬件算力需求，可适配服务器、边缘端GPU及其他服务器端AI加速卡。
  - 发布边缘端和CPU端超轻量SOTA目标检测模型[PP-PicoDet增强版](configs/picodet)，精度提升2%左右，CPU预测速度提升63%，新增参数量0.7M的PicoDet-XS模型，提供模型稀疏化和量化功能，便于模型加速，各类硬件无需单独开发后处理模块，降低部署门槛。
  - 发布实时行人分析工具[PP-Human](deploy/pipeline)，支持行人跟踪、人流量统计、人体属性识别与摔倒检测四大能力，基于真实场景数据特殊优化，精准识别各类摔倒姿势，适应不同环境背景、光线及摄像角度。
  - 新增[YOLOX](configs/yolox)目标检测模型，支持nano/tiny/s/m/l/x版本，x版本COCO val2017数据集精度51.8%。

- [更多版本发布](https://github.com/PaddlePaddle/PaddleDetection/releases)


## <img title="" src="https://user-images.githubusercontent.com/48054808/157795569-9fc77c85-732f-4870-9be0-99a7fe2cff27.png" alt="" width="20"> 简介

**PaddleDetection**为基于飞桨PaddlePaddle的端到端目标检测套件，内置**30+模型算法**及**250+预训练模型**，覆盖**目标检测、实例分割、跟踪、关键点检测**等方向，其中包括**服务器端和移动端高精度、轻量级**产业级SOTA模型、冠军方案和学术前沿算法，并提供配置化的网络模块组件、十余种数据增强策略和损失函数等高阶优化支持和多种部署方案，在打通数据处理、模型开发、训练、压缩、部署全流程的基础上，提供丰富的案例及教程，加速算法产业落地应用。

<div  align="center">
  <img src="https://user-images.githubusercontent.com/48054808/157826886-2e101a71-25a2-42f5-bf5e-30a97be28f46.gif" width="800"/>
</div>


## <img title="" src="https://user-images.githubusercontent.com/48054808/157800467-2a9946ad-30d1-49a9-b9db-ba33413d9c90.png" alt="" width="20"> 技术交流

- 如果你发现任何PaddleDetection存在的问题或者是建议, 欢迎通过[GitHub Issues](https://github.com/PaddlePaddle/PaddleDetection/issues)给我们提issues。

- **欢迎加入PaddleDetection 微信用户群（扫码填写问卷即可入群）**
  - **入群福利 💎：获取PaddleDetection团队整理的重磅学习大礼包🎁**
    - 📊 福利一：获取飞桨联合业界企业整理的开源数据集
    - 👨‍🏫 福利二：获取PaddleDetection历次发版直播视频与最新直播咨询
    - 🗳 福利三：获取垂类场景预训练模型集合，包括工业、安防、交通等5+行业场景
    - 🗂 福利四：获取10+全流程产业实操范例，覆盖火灾烟雾检测、人流量计数等产业高频场景
  <div align="center">
  <img src="https://user-images.githubusercontent.com/34162360/177678712-4655747d-4290-4ad9-b7a1-4564a5418ac6.jpg"  width = "200" />  
  </div>

## <img src="https://user-images.githubusercontent.com/48054808/157827140-03ffaff7-7d14-48b4-9440-c38986ea378c.png" width="20"/> 套件结构概览

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Architectures</b>
      </td>
      <td>
        <b>Backbones</b>
      </td>
      <td>
        <b>Components</b>
      </td>
      <td>
        <b>Data Augmentation</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
        <details><summary><b>Object Detection</b></summary>
          <ul>
            <li>YOLOv3</li>  
            <li>YOLOv5</li>  
            <li>YOLOv6</li>  
            <li>YOLOv7</li>  
            <li>PP-YOLOv1/v2</li>
            <li>PP-YOLO-Tiny</li>
            <li>PP-YOLOE</li>
            <li>PP-YOLOE+</li>
            <li>YOLOX</li>
         </ul></details>
      </ul>
      </td>
      <td>
        <details><summary><b>Details</b></summary>
        <ul>
          <li>ResNet(&vd)</li>
          <li>CSPResNet</li>
          <li>DarkNet</li>
          <li>CSPDarkNet</li>
          <li>ConvNeXt</li>
        </ul></details>
      </td>
      <td>
        <details><summary><b>Common</b></summary>
          <ul>
            <li>Sync-BN</li>
            <li>Group Norm</li>
            <li>DCNv2</li>
            <li>EMA</li>
          </ul> </details>
        </ul>
        <details><summary><b>FPN</b></summary>
          <ul>
            <li>CSP-PAN</li>
            <li>Custom-PAN</li>
            <li>ES-PAN</li>
          </ul> </details>
        </ul>  
        <details><summary><b>Loss</b></summary>
          <ul>
            <li>Smooth-L1</li>
            <li>GIoU/DIoU/CIoU</li>  
            <li>IoUAware</li>
            <li>Focal Loss</li>
            <li>VariFocal Loss</li>
          </ul> </details>
        </ul>  
        <details><summary><b>Post-processing</b></summary>
          <ul>
            <li>SoftNMS</li>
            <li>MatrixNMS</li>  
          </ul> </details>  
        </ul>
        <details><summary><b>Speed</b></summary>
          <ul>
            <li>FP16 training</li>
            <li>Multi-machine training </li>  
          </ul> </details>  
        </ul>  
      </td>
      <td>
        <details><summary><b>Details</b></summary>
        <ul>
          <li>Resize</li>  
          <li>Lighting</li>  
          <li>Flipping</li>  
          <li>Expand</li>
          <li>Crop</li>
          <li>Color Distort</li>  
          <li>Random Erasing</li>  
          <li>Mixup </li>
          <li>AugmentHSV</li>
          <li>Mosaic</li>
          <li>Cutmix </li>
          <li>Grid Mask</li>
          <li>Auto Augment</li>  
          <li>Random Perspective</li>  
        </ul> </details>  
      </td>  
    </tr>

</td>
    </tr>
  </tbody>
</table>

## <img src="https://user-images.githubusercontent.com/48054808/157801371-9a9a8c65-1690-4123-985a-e0559a7f9494.png" width="20"/> 模型性能概览

<details>
<summary><b> 云端模型性能对比</b></summary>

各模型结构和骨干网络的代表模型在COCO数据集上精度mAP和单卡Tesla V100上预测速度(FPS)对比图。

<div align="center">
  <img src="docs/images/fps_map.png" />
</div>

**说明：**

- `PP-YOLOE`是对`PP-YOLO v2`模型的进一步优化，在COCO数据集精度51.6%，Tesla V100预测速度78.1FPS
- `PP-YOLOE+`是对`PPOLOE`模型的进一步优化，在COCO数据集精度53.3%，Tesla V100预测速度78.1FPS
- 图中模型均可在[模型库](#模型库)中获取

</details>

<details>
<summary><b> 移动端模型性能对比</b></summary>

各移动端模型在COCO数据集上精度mAP和高通骁龙865处理器上预测速度(FPS)对比图。

<div align="center">
  <img src="docs/images/mobile_fps_map.png" width=600/>
</div>

**说明：**

- 测试数据均使用高通骁龙865(4\*A77 + 4\*A55)处理器batch size为1, 开启4线程测试，测试使用NCNN预测库，测试脚本见[MobileDetBenchmark](https://github.com/JiweiMaster/MobileDetBenchmark)
- [PP-PicoDet](configs/picodet)及[PP-YOLO-Tiny](configs/ppyolo)为PaddleDetection自研模型，其余模型PaddleDetection暂未提供

</details>

## <img src="https://user-images.githubusercontent.com/48054808/157829890-a535b8a6-631c-4c87-b861-64d4b32b2d6a.png" width="20"/> 模型库

<details>
<summary><b> 1. 通用检测</b></summary>

#### [PP-YOLOE+](./configs/ppyoloe)系列 推荐场景：Nvidia V100, T4等云端GPU和Jetson系列等边缘端设备

| 模型名称       | COCO精度（mAP） | V100 TensorRT FP16速度(FPS) | 配置文件                                                  | 模型下载                                                                                 |
|:---------- |:-----------:|:-------------------------:|:-----------------------------------------------------:|:------------------------------------------------------------------------------------:|
| PP-YOLOE+_s | 43.9        | 333.3                     | [链接](configs/ppyoloe/ppyoloe_plus_crn_s_80e_coco.yml)     | [下载地址](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_s_80e_coco.pdparams)      |
| PP-YOLOE+_m | 50.0        | 208.3                     | [链接](configs/ppyoloe/ppyoloe_plus_crn_m_80e_coco.yml)     | [下载地址](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_m_80e_coco.pdparams)     |
| PP-YOLOE+_l | 53.3        | 149.2                     | [链接](configs/ppyoloe/ppyoloe_plus_crn_l_80e_coco.yml) | [下载地址](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_m_80e_coco.pdparams) |
| PP-YOLOE+_x | 54.9        | 95.2                      | [链接](configs/ppyoloe/ppyoloe_plus_crn_x_80e_coco.yml) | [下载地址](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_x_80e_coco.pdparams) |

#### 前沿检测算法

| 模型名称                                                               | COCO精度（mAP） | V100 TensorRT FP16速度(FPS) | 配置文件                                                                                                         | 模型下载                                                                       |
|:------------------------------------------------------------------ |:-----------:|:-------------------------:|:------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------:|
| [YOLOX-l](configs/yolox)                                           | 50.1        | 107.5                     | [链接](configs/yolox/yolox_l_300e_coco.yml)                                                                    | [下载地址](https://paddledet.bj.bcebos.com/models/yolox_l_300e_coco.pdparams)  |
| [YOLOv5-l](configs/yolov5) | 48.6        | 136.0                     | [链接](configs/yolov5/yolov5_l_300e_coco.yml) | [下载地址](https://paddledet.bj.bcebos.com/models/yolov5_l_300e_coco.pdparams) |
| [YOLOv7-l](configs/yolov7) | 51.0        | 135.0                     | [链接](configs/yolov7/yolov7_l_300e_coco.yml) | [下载地址](https://paddledet.bj.bcebos.com/models/yolov7_l_300e_coco.pdparams) |

</details>


## <img src="https://user-images.githubusercontent.com/48054808/157828296-d5eb0ccb-23ea-40f5-9957-29853d7d13a9.png" width="20"/> 文档教程

### 入门教程

- [安装说明](docs/tutorials/INSTALL_cn.md)
- [快速体验](docs/tutorials/QUICK_STARTED_cn.md)
- [数据准备](docs/tutorials/data/README.md)
- [PaddleDetection全流程使用](docs/tutorials/GETTING_STARTED_cn.md)
- [FAQ/常见问题汇总](docs/tutorials/FAQ)

### 进阶教程

- 参数配置

  - [PP-YOLO参数说明](docs/tutorials/config_annotation/ppyolo_r50vd_dcn_1x_coco_annotation.md)

- 模型压缩(基于[PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim))

  - [剪裁/量化/蒸馏教程](configs/slim)

- [推理部署](deploy/README.md)

  - [模型导出教程](deploy/EXPORT_MODEL.md)
  - [Paddle Inference部署](deploy/README.md)
    - [Python端推理部署](deploy/python)
    - [C++端推理部署](deploy/cpp)
  - [Paddle-Lite部署](deploy/lite)
  - [Paddle Serving部署](deploy/serving)
  - [ONNX模型导出](deploy/EXPORT_ONNX_MODEL.md)
  - [推理benchmark](deploy/BENCHMARK_INFER.md)

- 进阶开发

  - [数据处理模块](docs/advanced_tutorials/READER.md)
  - [新增检测模型](docs/advanced_tutorials/MODEL_TECHNICAL.md)
  - 二次开发教程
    - [目标检测](docs/advanced_tutorials/customization/detection.md)


## <img src="https://user-images.githubusercontent.com/48054808/157835981-ef6057b4-6347-4768-8fcc-cd07fcc3d8b0.png" width="20"/> 版本更新

版本更新内容请参考[版本更新文档](docs/CHANGELOG.md)

## <img src="https://user-images.githubusercontent.com/48054808/157835276-9aab9d1c-1c46-446b-bdd4-5ab75c5cfa48.png" width="20"/> 引用

```
@misc{ppdet2019,
title={PaddleDetection, Object detection and instance segmentation toolkit based on PaddlePaddle.},
author={PaddlePaddle Authors},
howpublished = {\url{https://github.com/PaddlePaddle/PaddleDetection}},
year={2019}
}
```
