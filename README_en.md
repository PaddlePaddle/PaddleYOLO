[简体中文](README_cn.md) | English

## Introduction

**PaddleYOLO** is a YOLO series toolbox based on [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection), **only relevant codes of YOLO series models are included**. It supports `YOLOv3`,`PP-YOLO`,`PP-YOLOv2`,`PP-YOLOE`,`PP-YOLOE+`,`YOLOX`,`YOLOv5`,`YOLOv6`,`YOLOv7`,`YOLOv8`,`YOLOv5u`,`YOLOv7u`,`RTMDet` and so on, see COCO dataset ModelZoo in [ModelZoo](docs/MODEL_ZOO_en.md) and [configs](configs/).

<div  align="center">
  <img src="https://user-images.githubusercontent.com/13104100/213197403-c8257486-9ac4-486f-a0d5-4e3fe27ca852.jpg" width="480"/>
  <img src="https://user-images.githubusercontent.com/13104100/213197635-eeb55433-bb2d-44f6-b374-73c616cfab24.jpg" width="480"/>
</div>

**Notes：**
 - The Licence of **PaddleYOLO** is **[GPL 3.0](LICENSE)**, the codes of [YOLOv5](configs/yolov5),[YOLOv6](configs/yolov6),[YOLOv7](configs/yolov7) and [YOLOv8](configs/yolov8) will not be merged into [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection). Except for these YOLO models, other YOLO models are recommended to use in [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection), **which will be the first to release the latest progress of PP-YOLO series detection model**;
 - To use **PaddleYOLO**, **PaddlePaddle-2.3.2 or above is recommended**，please refer to the [official website](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html) to download the appropriate version. **For Windows platforms, please install the paddle develop version**;
 - **PaddleYOLO's [Roadmap](https://github.com/PaddlePaddle/PaddleYOLO/issues/44)** issue collects feature requests from user, welcome to put forward any opinions and suggestions.

## Tutorials

<details open>
<summary>Install</summary>

Clone repo and install [requirements.txt](./requirements.txt) in a
[**Python>=3.7.0**](https://www.python.org/) environment, including
[**PaddlePaddle>=2.3.2**](https://www.paddlepaddle.org.cn/install/).

```bash
git clone https://github.com/PaddlePaddle/PaddleYOLO  # clone
cd PaddleYOLO
pip install -r requirements.txt  # install
```

</details>


<details open>
<summary>Training/Evaluation/Inference</summary>

Write the following commands in a script file, such as ```run.sh```, and run as：```sh run.sh```. You can also run the command line sentence by sentence.

```bash
model_name=ppyoloe # yolov7
job_name=ppyoloe_plus_crn_s_80e_coco # yolov7_tiny_300e_coco

config=configs/${model_name}/${job_name}.yml
log_dir=log_dir/${job_name}
# weights=https://bj.bcebos.com/v1/paddledet/models/${job_name}.pdparams
weights=output/${job_name}/model_final.pdparams

# 1.training（single GPU / multi GPU）
# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c ${config} --eval --amp
python -m paddle.distributed.launch --log_dir=${log_dir} --gpus 0,1,2,3,4,5,6,7 tools/train.py -c ${config} --eval --amp

# 2.eval
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c ${config} -o weights=${weights} --classwise

# 3.infer
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c ${config} -o weights=${weights} --infer_img=demo/000000014439_640x640.jpg --draw_threshold=0.5
# CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c ${config} -o weights=${weights} --infer_dir=demo/ --draw_threshold=0.5
```

</details>


<details>
<summary>Deployment/Speed</summary>

Write the following commands in a script file, such as ```run.sh```, and run as：```sh run.sh```. You can also run the command line sentence by sentence.

```bash
model_name=ppyoloe # yolov7
job_name=ppyoloe_plus_crn_s_80e_coco # yolov7_tiny_300e_coco

config=configs/${model_name}/${job_name}.yml
log_dir=log_dir/${job_name}
# weights=https://bj.bcebos.com/v1/paddledet/models/${job_name}.pdparams
weights=output/${job_name}/model_final.pdparams

# 4.export
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c ${config} -o weights=${weights} # trt=True

# CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c ${config} -o weights=${weights} exclude_post_process=True # trt=True

# CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c ${config} -o weights=${weights} exclude_nms=True # trt=True

# 5.deploy infer
CUDA_VISIBLE_DEVICES=0 python deploy/python/infer.py --model_dir=output_inference/${job_name} --image_file=demo/000000014439_640x640.jpg --device=GPU

# 6.deploy speed, add '--run_mode=trt_fp16' to test in TensorRT FP16 mode
CUDA_VISIBLE_DEVICES=0 python deploy/python/infer.py --model_dir=output_inference/${job_name} --image_file=demo/000000014439_640x640.jpg --device=GPU --run_benchmark=True # --run_mode=trt_fp16

# 7.export onnx
paddle2onnx --model_dir output_inference/${job_name} --model_filename model.pdmodel --params_filename model.pdiparams --opset_version 12 --save_file ${job_name}.onnx

# 8.onnx speed
/usr/local/TensorRT-8.0.3.4/bin/trtexec --onnx=${job_name}.onnx --workspace=4096 --avgRuns=10 --shapes=input:1x3x640x640 --fp16
/usr/local/TensorRT-8.0.3.4/bin/trtexec --onnx=${job_name}.onnx --workspace=4096 --avgRuns=10 --shapes=input:1x3x640x640 --fp32
```

**Note：**
- If you want to switch models, just modify the first two lines, such as:
  ```
  model_name=yolov7
  job_name=yolov7_tiny_300e_coco
  ```
- For **exporting onnx**, you should install [Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX) by `pip install paddle2onnx` at first.
- For **FLOPs(G) and Params(M)**, you should install [PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim) by `pip install paddleslim` at first, then set `print_flops: True` and `print_params: True` in [runtime.yml](configs/runtime.yml). Make sure **single scale** like 640x640, **MACs are printed，FLOPs=2*MACs**.

</details>


<details open>
<summary> [Training Custom dataset](https://github.com/PaddlePaddle/PaddleYOLO/issues/43) </summary>

- Please refer to [doc](docs/MODEL_ZOO_en.md#CustomDataset) and [issue](https://github.com/PaddlePaddle/PaddleYOLO/issues/43).
- PaddleDetection team provides various **feature detection models based on PP-YOLOE** , which can also be used as a reference to modify on your custom dataset. Please refer to [PP-YOLOE application](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/ppyoloe/application), [pphuman](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/pphuman), [ppvehicle](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/ppvehicle), [visdrone](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/visdrone) and [smalldet](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/smalldet).
- PaddleDetection also provides **various YOLO models  for VOC dataset** , which can also be used as a reference to modify on your custom dataset. Please refer to [voc](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.6/configs/voc).
- Please **ensure the corresponding COCO trained weights are loaded as pre-train weights** at first. Set the `pretrain_weights: ` with corresponding COCO trained weights in the config file, and it will generally prompt that the number of channels convolved by the head classification layer does not correspond, which is a normal phenomenon, because the number of types of user-defined data sets is generally inconsistent with that of COCO data sets.
- We recommend to use YOLO detection model **with a total `batch_size` at least greater than `64` to train**. If the resources are insufficient, please **use the smaller model** or **reduce the input size of the model**. To ensure high detection accuracy, **you'd better not try to using single GPU or total `batch_size` less than `64` for training**;

</details>


## Updates

* 【2023/03/13】Support [YOLOv5u](configs/yolov5/yolov5u) and [YOLOv7u](configs/yolov7/yolov7u) inference and deploy;
* 【2023/01/10】Support [YOLOv8](configs/yolov8) inference and deploy;
* 【2022/09/29】Support [RTMDet](configs/rtmdet) inference and deploy;
* 【2022/09/26】Release [PaddleYOLO](https://github.com/PaddlePaddle/PaddleYOLO), see [ModelZoo](docs/MODEL_ZOO_en.md);
* 【2022/09/19】Support the new version of [YOLOv6](configs/yolov6), including n/t/s/m/l model;
* 【2022/08/23】Release `YOLOSeries` codebase: support `YOLOv3`,`PP-YOLOE`,`PP-YOLOE+`,`YOLOX`,`YOLOv5`,`YOLOv6` and `YOLOv7`; support using `ConvNeXt` backbone to get high-precision version of `PP-YOLOE`,`YOLOX` and `YOLOv5`; support PaddleSlim accelerated quantitative training `PP-YOLOE`,`YOLOv5`,`YOLOv6` and `YOLOv7`. For details, please read this [article](https://mp.weixin.qq.com/s/Hki01Zs2lQgvLSLWS0btrA)；


## <img src="https://user-images.githubusercontent.com/48054808/157793354-6e7f381a-0aa6-4bb7-845c-9acf2ecc05c3.png" width="20"/> Product Update

- 🔥 **2023.3.14：Release PaddleYOLO [release/2.6](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.6)**
  - 💡 Model kit：
    - Support `YOLOv8`,`YOLOv5u`,`YOLOv7u` inference and deploy.
    - Support `Swin-Transformer`、`ViT`、`FocalNet` backbone to get high-precision version of `PP-YOLOE+`.
    - Support `YOLOv8` in [FastDeploy](https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/vision/detection/paddledetection).

- 🔥 **2022.9.26：Release PaddleYOLO [release/2.5](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5)**
  - 💡 Model kit：
    - Release [PaddleYOLO](https://github.com/PaddlePaddle/PaddleYOLO): support `YOLOv3`,`PP-YOLOE`,`PP-YOLOE+`,`YOLOX`,`YOLOv5`,`YOLOv6` and `YOLOv7`; support using `ConvNeXt` backbone to get high-precision version of `PP-YOLOE`,`YOLOX` and `YOLOv5`; support PaddleSlim accelerated quantitative training `PP-YOLOE`,`YOLOv5`,`YOLOv6` and `YOLOv7`.

- 🔥 **2022.8.26：PaddleDetection [release/2.5 version](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5)**

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
            <li>YOLOv8</li>  
            <li>PP-YOLOv1/v2</li>
            <li>PP-YOLO-Tiny</li>
            <li>PP-YOLOE</li>
            <li>PP-YOLOE+</li>
            <li>YOLOX</li>
            <li>RTMDet</li>
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
          <li>EfficientRep</li>
          <li>CSPBepBackbone</li>
          <li>ELANNet</li>
          <li>CSPNeXt</li>
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
            <li>YOLOv3FPN</li>
            <li>PPYOLOFPN</li>
            <li>PPYOLOTinyFPN</li>
            <li>PPYOLOPAN</li>
            <li>YOLOCSPPAN</li>
            <li>Custom-PAN</li>
            <li>RepPAN</li>
            <li>CSPRepPAN</li>
            <li>ELANFPN</li>
            <li>ELANFPNP6</li>
            <li>CSPNeXtPAFPN</li>
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

## <img title="" src="https://user-images.githubusercontent.com/48054808/157835345-f5d24128-abaf-4813-b793-d2e5bdc70e5a.png" alt="" width="20">  License

PaddleYOLO is provided under the [GPL-3.0 license](LICENSE)

## <img src="https://user-images.githubusercontent.com/48054808/157835276-9aab9d1c-1c46-446b-bdd4-5ab75c5cfa48.png" width="20"/> Quote

```
@misc{ppdet2019,
title={PaddleDetection, Object detection and instance segmentation toolkit based on PaddlePaddle.},
author={PaddlePaddle Authors},
howpublished = {\url{https://github.com/PaddlePaddle/PaddleDetection}},
year={2019}
}
```
