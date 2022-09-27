[简体中文](README_cn.md) | English

## Introduction

**PaddleYOLO** is a YOLO Series toolbox based on [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection), **only relevant codes of YOLO series models are included**. It supports `YOLOv3`,`PP-YOLO`,`PP-YOLOv2`,`PP-YOLOE`,`PP-YOLOE+`,`YOLOX`,`YOLOv5`,`YOLOv6`,`YOLOv7` and so on, see [ModelZoo](docs/MODEL_ZOO_en.md);

## Updates

* 【2022/09/26】Release [PaddleYOLO](https://github.com/PaddlePaddle/PaddleYOLO), see [ModelZoo](docs/MODEL_ZOO_en.md);
* 【2022/09/19】Support the new version of [YOLOv6](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov6), including n/t/s/m/l model;
* 【2022/08/23】Release `YOLOSeries` codebase: support `YOLOv3`,`PP-YOLOE`,`PP-YOLOE+`,`YOLOX`,`YOLOv5`,`YOLOv6` and `YOLOv7`; support using `ConvNeXt` backbone to get high-precision version of `PP-YOLOE`,`YOLOX` and `YOLOv5`; support PaddleSlim accelerated quantitative training `PP-YOLOE`,`YOLOv5`,`YOLOv6` and `YOLOv7`. For details, please read this [article](https://mp.weixin.qq.com/s/Hki01Zs2lQgvLSLWS0btrA)；


**Notes：**
 - The Licence of **PaddleYOLO** is **GPL 3.0**, the codes of [YOLOv5](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov5),[YOLOv7](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov7) and [YOLOv6](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov6) will not be merged into [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection). Except for these three YOLO models, other YOLO models are recommended to use in [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection), **which will be the first to release the latest progress of PP-YOLO series detection model**;
 - To use **PaddleYOLO**, **PaddlePaddle-2.3.2 or above is recommended**，please refer to the [official website](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html) to download the appropriate version. **For Windows platforms, please install the paddle develop version**;


## <img src="https://user-images.githubusercontent.com/48054808/157793354-6e7f381a-0aa6-4bb7-845c-9acf2ecc05c3.png" width="20"/> Product Update

- 🔥 **2022.9.26：Release PaddleYOLO[release/2.5](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5)**
  - 💡 Model kit：
    - Release [PaddleYOLO](https://github.com/PaddlePaddle/PaddleYOLO): support `YOLOv3`,`PP-YOLOE`,`PP-YOLOE+`,`YOLOX`,`YOLOv5`,`YOLOv6` and `YOLOv7`; support using `ConvNeXt` backbone to get high-precision version of `PP-YOLOE`,`YOLOX` and `YOLOv5`; support PaddleSlim accelerated quantitative training `PP-YOLOE`,`YOLOv5`,`YOLOv6` and `YOLOv7`.

- 🔥 **2022.8.26：PaddleDetection releases[release/2.5 version](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5)**

  - 🗳 Model features：

    - Release [PP-YOLOE+](configs/ppyoloe): Increased accuracy by a maximum of 2.4% mAP to 54.9% mAP, 3.75 times faster model training convergence rate, and up to 2.3 times faster end-to-end inference speed; improved generalization for multiple downstream tasks
    - Release [PicoDet-NPU](configs/picodet) model which supports full quantization deployment of models; add [PicoDet](configs/picodet) layout analysis model
    - Release [PP-TinyPose Plus](./configs/keypoint/tiny_pose/). With 9.1% AP accuracy improvement in physical exercise, dance, and other scenarios, our PP-TinyPose Plus supports unconventional movements such as turning to one side, lying down, jumping, and high lifts

  - 🔮 Functions in different scenarios

    - Release the pedestrian analysis tool [PP-Human v2](./deploy/pipeline). It introduces four new behavior recognition: fighting, telephoning, smoking, and trespassing. The underlying algorithm performance is optimized, covering three core algorithm capabilities: detection, tracking, and attributes of pedestrians. Our model provides end-to-end development and model optimization strategies for beginners and supports online video streaming input.
    - First release [PP-Vehicle](./deploy/pipeline), which has four major functions: license plate recognition, vehicle attribute analysis (color, model), traffic flow statistics, and violation detection. It is compatible with input formats, including pictures, online video streaming, and video. And we also offer our users a comprehensive set of tutorials for customization.

  - 💡 Cutting-edge algorithms：

    - Covers [YOLO family](https://github.com/PaddlePaddle/PaddleYOLO) classic and latest models: YOLOv3, PP-YOLOE (a real-time high-precision object detection model developed by Baidu PaddlePaddle), and cutting-edge detection algorithms such as YOLOv4, YOLOv5, YOLOX, YOLOv6, and YOLOv7
    - Newly add high precision detection model based on [ViT](configs/vitdet) backbone network, with a 55.7% mAP accuracy on COCO dataset; newly add multi-object tracking model [OC-SORT](configs/mot/ocsort); newly add [ConvNeXt](configs/convnext) backbone network.

  - 📋 Industrial applications: Newly add [Smart Fitness](https://aistudio.baidu.com/aistudio/projectdetail/4385813), [Fighting recognition](https://aistudio.baidu.com/aistudio/projectdetail/4086987?channelType=0&channel=0),[ and Visitor Analysis](https://aistudio.baidu.com/aistudio/projectdetail/4230123?channelType=0&channel=0).

- 2022.3.24：PaddleDetection released[release/2.4 version](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4)  
  - Release high-performanace SOTA object detection model [PP-YOLOE](configs/ppyoloe). It integrates cloud and edge devices and provides S/M/L/X versions. In particular, Verson L has the accuracy as 51.4% on COCO test 2017 dataset, inference speed as 78.1 FPS on a single Test V100. It supports mixed precision training, 33% faster than PP-YOLOv2. Its full range of multi-sized models can meet different hardware arithmetic requirements, and adaptable to server, edge-device GPU and other AI accelerator cards on servers.
  - Release ultra-lightweight SOTA object detection model [PP-PicoDet Plus](configs/picodet) with 2% improvement in accuracy and 63% improvement in CPU inference speed. Add PicoDet-XS model with a 0.7M parameter, providing model sparsification and quantization functions for model acceleration. No specific post processing module is required for all the hardware, simplifying the deployment.  
  - Release the real-time pedestrian analysis tool [PP-Human](deploy/pphuman). It has four major functions: pedestrian tracking, visitor flow statistics, human attribute recognition and falling detection. For falling detection, it is optimized based on real-life data with accurate recognition of various types of falling posture. It can adapt to different environmental background, light and camera angle.
  - Add [YOLOX](configs/yolox) object detection model with nano/tiny/S/M/L/X. X version has the accuracy as 51.8% on COCO  Val2017 dataset.

- [More releases](https://github.com/PaddlePaddle/PaddleDetection/releases)

## <img title="" src="https://user-images.githubusercontent.com/48054808/157795569-9fc77c85-732f-4870-9be0-99a7fe2cff27.png" alt="" width="20"> Brief Introduction

**PaddleDetection** is an end-to-end object detection development kit based on PaddlePaddle. Providing **over 30 model algorithm** and **over 250 pre-trained models**, it covers object detection, instance segmentation, keypoint detection, multi-object tracking. In particular, PaddleDetection offers **high- performance & light-weight** industrial SOTA models on **servers and mobile** devices, champion solution and cutting-edge algorithm. PaddleDetection provides various data augmentation methods, configurable network components, loss functions and other advanced optimization & deployment schemes. In addition to running through the whole process of data processing, model development, training, compression and deployment, PaddlePaddle also provides rich cases and tutorials to accelerate the industrial application of algorithm.

<div  align="center">
  <img src="https://user-images.githubusercontent.com/22989727/189122825-ee1c1db2-b5f9-42c0-88b4-7975e1ec239d.gif" width="800"/>
</div>

## <img src="https://user-images.githubusercontent.com/48054808/157799599-e6a66855-bac6-4e75-b9c0-96e13cb9612f.png" width="20"/> Features

- **Rich model library**: PaddleDetection provides over 250 pre-trained models including **object detection, instance segmentation, face recognition, multi-object tracking**. It covers a variety of **global competition champion** schemes.
- **Simple to use**: Modular design, decoupling each network component, easy for developers to build and try various detection models and optimization strategies, quick access to high-performance, customized algorithm.
- **Getting Through End to End**: PaddlePaddle gets through end to end from data augmentation, constructing models, training, compression, depolyment. It also supports multi-architecture, multi-device deployment for **cloud and edge** device.
- **High Performance**: Due to the high performance core, PaddlePaddle has clear advantages in training speed and memory occupation. It also supports FP16 training and multi-machine training.

<div  align="center">
  <img src="https://user-images.githubusercontent.com/22989727/189066615-89d1dde2-54bc-4946-887e-fce50069206e.png" width="800"/>
</div>

## <img title="" src="https://user-images.githubusercontent.com/48054808/157800467-2a9946ad-30d1-49a9-b9db-ba33413d9c90.png" alt="" width="20"> Exchanges

- If you have any question or suggestion, please give us your valuable input via [GitHub Issues](https://github.com/PaddlePaddle/PaddleDetection/issues)

  Welcome to join PaddleDetection user groups on WeChat (scan the QR code, add and reply "D" to the assistant)

  <div align="center">
  <img src="https://user-images.githubusercontent.com/34162360/177678712-4655747d-4290-4ad9-b7a1-4564a5418ac6.jpg"  width = "200" />  
  </div>

## <img src="https://user-images.githubusercontent.com/48054808/157827140-03ffaff7-7d14-48b4-9440-c38986ea378c.png" width="20"/> Kit Structure

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
        <details open><summary><b>Object Detection</b></summary>
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
        <details open><summary><b>Details</b></summary>
        <ul>
          <li>ResNet(&vd)</li>
          <li>CSPResNet</li>
          <li>DarkNet</li>
          <li>CSPDarkNet</li>
          <li>ConvNeXt</li>
        </ul></details>
      </td>
      <td>
        <details open><summary><b>Common</b></summary>
          <ul>
            <li>Sync-BN</li>
            <li>Group Norm</li>
            <li>DCNv2</li>
            <li>EMA</li>
          </ul> </details>
        </ul>
        <details open><summary><b>FPN</b></summary>
          <ul>
            <li>CSP-PAN</li>
            <li>Custom-PAN</li>
            <li>ES-PAN</li>
          </ul> </details>
        </ul>  
        <details open><summary><b>Loss</b></summary>
          <ul>
            <li>Smooth-L1</li>
            <li>GIoU/DIoU/CIoU</li>  
            <li>IoUAware</li>
            <li>Focal Loss</li>
            <li>VariFocal Loss</li>
          </ul> </details>
        </ul>  
        <details open><summary><b>Post-processing</b></summary>
          <ul>
            <li>SoftNMS</li>
            <li>MatrixNMS</li>  
          </ul> </details>  
        </ul>
        <details open><summary><b>Speed</b></summary>
          <ul>
            <li>FP16 training</li>
            <li>Multi-machine training </li>  
          </ul> </details>  
        </ul>  
      </td>
      <td>
        <details open><summary><b>Details</b></summary>
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

## <img src="https://user-images.githubusercontent.com/48054808/157801371-9a9a8c65-1690-4123-985a-e0559a7f9494.png" width="20"/> Model Performance

<details>
<summary><b> Performance comparison of Cloud models</b></summary>

The comparison between COCO mAP and FPS on Tesla V100 of representative models of each architectures and backbones.

<div align="center">
  <img src="docs/images/fps_map.png" />
</div>

**Clarification：**

- `PP-YOLOE` are optimized `PP-YOLO v2`. It reached accuracy as 51.4% on COCO dataset, inference speed as 78.1 FPS on Tesla V100
- `PP-YOLOE+` are optimized `PP-YOLOE`. It reached accuracy as 53.3% on COCO dataset, inference speed as 78.1 FPS on Tesla V100
- The models in the figure are available in the[ model library](#模型库)

</details>

<details>
<summary><b> Performance omparison on mobiles</b></summary>

The comparison between COCO mAP and FPS on Qualcomm Snapdragon 865 processor of models on mobile devices.

<div align="center">
  <img src="docs/images/mobile_fps_map.png" width=600/>
</div>

**Clarification：**

- Tests were conducted on Qualcomm Snapdragon 865 (4 \*A77 + 4 \*A55) batch_size=1, 4 thread, and NCNN inference library, test script see [MobileDetBenchmark](https://github.com/JiweiMaster/MobileDetBenchmark)
- [PP-PicoDet](configs/picodet) and [PP-YOLO-Tiny](configs/ppyolo) are self-developed models of PaddleDetection, and other models are not tested yet.

</details>

## <img src="https://user-images.githubusercontent.com/48054808/157829890-a535b8a6-631c-4c87-b861-64d4b32b2d6a.png" width="20"/> Model libraries

<details>
<summary><b> 1. General detection</b></summary>

#### PP-YOLOE series Recommended scenarios: Cloud GPU such as Nvidia V100, T4 and edge devices such as Jetson series

| Model      | COCO Accuracy（mAP） | V100 TensorRT FP16 Speed(FPS) | Configuration                                           | Download                                                                                 |
|:---------- |:------------------:|:-----------------------------:|:-------------------------------------------------------:|:----------------------------------------------------------------------------------------:|
| PP-YOLOE+_s | 43.9        | 333.3                     | [link](configs/ppyoloe/ppyoloe_plus_crn_s_80e_coco.yml)     | [download](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_s_80e_coco.pdparams)      |
| PP-YOLOE+_m | 50.0        | 208.3                     | [link](configs/ppyoloe/ppyoloe_plus_crn_m_80e_coco.yml)     | [download](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_m_80e_coco.pdparams)     |
| PP-YOLOE+_l | 53.3        | 149.2                     | [link](configs/ppyoloe/ppyoloe_plus_crn_l_80e_coco.yml) | [download](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_m_80e_coco.pdparams) |
| PP-YOLOE+_x | 54.9        | 95.2                      | [link](configs/ppyoloe/ppyoloe_plus_crn_x_80e_coco.yml) | [download](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_x_80e_coco.pdparams) |

#### Frontier detection algorithm

| Model    | COCO Accuracy（mAP） | V100 TensorRT FP16 speed(FPS) | Configuration                                                                                                  | Download                                                                       |
|:-------- |:------------------:|:-----------------------------:|:--------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------:|
| [YOLOX-l](configs/yolox)  | 50.1               | 107.5                         | [Link](configs/yolox/yolox_l_300e_coco.yml)                                                                    | [Download](https://paddledet.bj.bcebos.com/models/yolox_l_300e_coco.pdparams)  |
| [YOLOv5-l](configs/yolov5) | 48.6               | 136.0                         | [Link](configs/yolov5/yolov5_l_300e_coco.yml) | [Download](https://paddledet.bj.bcebos.com/models/yolov5_l_300e_coco.pdparams) |
| [YOLOv7-l](configs/yolov7) | 51.0        | 135.0                     | [链接](configs/yolov7/yolov7_l_300e_coco.yml) | [下载地址](https://paddledet.bj.bcebos.com/models/yolov7_l_300e_coco.pdparams) |

</details>

## <img src="https://user-images.githubusercontent.com/48054808/157828296-d5eb0ccb-23ea-40f5-9957-29853d7d13a9.png" width="20"/>Document tutorials

### Introductory tutorials

- [Installation](docs/tutorials/INSTALL_cn.md)
- [Quick start](docs/tutorials/QUICK_STARTED_cn.md)
- [Data preparation](docs/tutorials/data/README.md)
- [Geting Started on PaddleDetection](docs/tutorials/GETTING_STARTED_cn.md)
- [FAQ]((docs/tutorials/FAQ)

### Advanced tutorials

- Configuration

  - [PP-YOLO Configuration](docs/tutorials/config_annotation/ppyolo_r50vd_dcn_1x_coco_annotation.md)

- Compression based on [PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim)

  - [Pruning/Quantization/Distillation Tutorial](configs/slim)

- [Inference deployment](deploy/README.md)

  - [Export model for inference](deploy/EXPORT_MODEL.md)

  - [Paddle Inference deployment](deploy/README.md)

    - [Inference deployment with Python](deploy/python)
    - [Inference deployment with C++](deploy/cpp)

  - [Paddle-Lite deployment](deploy/lite)

  - [Paddle Serving deployment](deploy/serving)

  - [ONNX model export](deploy/EXPORT_ONNX_MODEL.md)

  - [Inference benchmark](deploy/BENCHMARK_INFER.md)

- Advanced development

  - [Data processing module](docs/advanced_tutorials/READER.md)
  - [New object detection models](docs/advanced_tutorials/MODEL_TECHNICAL.md)
  - Custumization
    - [Object detection](docs/advanced_tutorials/customization/detection.md)


## <img src="https://user-images.githubusercontent.com/48054808/157835981-ef6057b4-6347-4768-8fcc-cd07fcc3d8b0.png" width="20"/> Version updates

Please refer to the[ Release note ](https://github.com/PaddlePaddle/Paddle/wiki/PaddlePaddle-2.3.0-Release-Note-EN)for more details about the updates


## <img src="https://user-images.githubusercontent.com/48054808/157835276-9aab9d1c-1c46-446b-bdd4-5ab75c5cfa48.png" width="20"/> Quote

```
@misc{ppdet2019,
title={PaddleDetection, Object detection and instance segmentation toolkit based on PaddlePaddle.},
author={PaddlePaddle Authors},
howpublished = {\url{https://github.com/PaddlePaddle/PaddleDetection}},
year={2019}
}
```
