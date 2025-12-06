# 🌾 水稻稻瘟病智能检测系统 | Rice Blast Disease Detection System

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PaddlePaddle-2.4+-green.svg" alt="PaddlePaddle">
  <img src="https://img.shields.io/badge/Flask-2.3+-red.svg" alt="Flask">
  <img src="https://img.shields.io/badge/YOLOv10-Detection-orange.svg" alt="YOLOv10">
</div>

## 📖 项目简介

这是一个基于深度学习的水稻稻瘟病智能检测系统，专为小学生科技创新竞赛设计。系统采用了先进的YOLOv10目标检测算法，能够快速准确地识别水稻叶片上的稻瘟病病斑，帮助农民及时发现和防治病害，保护粮食安全。

### ✨ 主要特性

- 🖼️ **单图检测**：支持上传单张图片进行病害检测
- 📁 **批量检测**：支持加载整个文件夹，批量检测多张图片
- 🔄 **便捷导航**：提供前后翻页功能，方便浏览检测结果
- 🎯 **精准识别**：基于YOLOv10算法，检测准确率高
- 🎨 **美观界面**：水稻主题UI设计，符合农业场景
- ⚙️ **灵活配置**：可调节置信度阈值，控制检测敏感度
- 📊 **详细结果**：显示病斑数量、位置和置信度信息

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PaddlePaddle 2.4+
- 其他依赖见 `requirements.txt`

### 安装步骤

1. **克隆项目**（如果还未克隆）

```bash
cd e:\projects\myPaddleYOLO
```

2. **安装依赖**

```bash
cd rice_demo
pip install -r requirements.txt
```

注意：
- 如果使用GPU，请确保安装了 `paddlepaddle-gpu`
- 如果使用CPU，请安装 `paddlepaddle` 并在 `requirements.txt` 中相应修改

3. **准备模型**

确保您已经训练好了模型，并知道模型配置文件和权重文件的路径。例如：
- 配置文件：`../configs/yolov10/rice_blast_yolov10_n.yml`
- 权重文件：`../output/rice_blast_yolov10_n/model_final.pdparams`

### 运行应用

```bash
python app.py --config ../configs/yolov8/yolov8_n_100e_16b8b_rice200.yml --weights ../output/yolov8_n_100e_16b8b_rice/best_model.pdparams
```

如果使用GPU：

```bash
python app.py --config ../configs/yolov10/rice_blast_yolov10_n.yml --weights ../output/rice_blast_yolov10_n/model_final.pdparams --use_gpu
```

指定端口和主机：

```bash
python app.py --config ../configs/yolov10/rice_blast_yolov10_n.yml --weights ../output/rice_blast_yolov10_n/model_final.pdparams --host 0.0.0.0 --port 8080
```

### 访问应用

在浏览器中打开：

```
http://127.0.0.1:5000
```

或者如果指定了其他主机和端口，使用相应的地址。

## 📝 使用说明

### 1. 单图检测模式

1. 点击"上传图片"区域或将图片拖拽到该区域
2. 图片将显示在右侧预览区
3. 调整置信度阈值（可选）
4. 点击"开始检测"按钮
5. 等待检测完成，查看结果

### 2. 批量检测模式

1. 在"加载文件夹"输入框中输入包含图片的文件夹路径
   - 例如：`e:\images\rice_samples`
2. 点击"加载"按钮
3. 系统会显示文件夹中的图片数量
4. 使用"上一张"/"下一张"按钮浏览图片
5. 对每张图片点击"开始检测"进行检测
6. 可以使用键盘左右箭头键快速切换图片

### 3. 查看检测结果

检测完成后，系统会显示：
- ✅ 病斑检测数量
- ✅ 当前图片名称
- ✅ 每个病斑的详细信息（类型、置信度、位置）
- ✅ 在图片上用红色框标注病斑位置

### 4. 快捷键

- `←` / `→`：切换上一张/下一张图片
- `Enter`：开始检测当前图片

## 🎨 界面设计

系统采用水稻主题设计：
- 🌾 **绿色主色调**：象征健康的水稻
- ✨ **金色点缀**：象征丰收的稻穗
- 🔴 **红色病斑标记**：醒目地标识病害位置
- 🎭 **动态背景**：摇曳的水稻动画，增加视觉趣味

## 🛠️ 技术架构

### 后端技术

- **Flask**：轻量级Web框架
- **PaddlePaddle**：百度深度学习框架
- **PaddleDetection**：目标检测工具库
- **YOLOv10**：最新的YOLO目标检测算法

### 前端技术

- **HTML5/CSS3**：现代化网页设计
- **JavaScript**：交互功能实现
- **Font Awesome**：图标库

### 项目结构

```
rice_demo/
├── app.py                 # Flask应用主文件
├── requirements.txt       # 依赖包列表
├── README.md             # 项目说明文档
├── templates/            # HTML模板
│   └── index.html       # 主页面
├── static/              # 静态资源
│   ├── css/
│   │   └── styles.css   # 样式文件
│   └── js/
│       └── script.js    # JavaScript脚本
└── uploads/             # 上传文件临时存储
```

## 🔧 配置说明

### 命令行参数

- `--config`：模型配置文件路径（必需）
- `--weights`：模型权重文件路径（必需）
- `--use_gpu`：是否使用GPU（可选，默认False）
- `--port`：服务器端口（可选，默认5000）
- `--host`：服务器主机（可选，默认127.0.0.1）

### 置信度阈值

在界面上可以调整置信度阈值（0.0-1.0）：
- 较低的阈值：检测更多的潜在病斑，可能有误报
- 较高的阈值：只显示高置信度的病斑，减少误报

## 📊 示例效果

检测结果示例：
- 检测到的病斑会用红色边框标注
- 显示病斑类别和置信度百分比
- 列出所有检测到的病斑详细信息

## ⚠️ 注意事项

1. **图片格式**：支持 JPG, JPEG, PNG, BMP 格式
2. **文件大小**：单个图片最大 16MB
3. **文件夹路径**：Windows系统使用完整路径，如 `e:\images\samples`
4. **模型文件**：确保模型已训练完成且路径正确
5. **GPU使用**：首次使用GPU需要安装CUDA和cuDNN

## 🤝 贡献

这是一个小学生科技创新竞赛项目，欢迎提出改进建议！

## 📄 许可证

本项目基于 Apache License 2.0 开源协议。

## 🎓 教育意义

通过这个项目，学生可以学习到：
- 🔬 人工智能在农业中的应用
- 💻 Web应用开发的基础知识
- 🌾 水稻病害防治的重要性
- 🤖 深度学习模型的实际应用

## 📮 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 发送邮件至项目维护者

---

<div align="center">
  <p>🌾 助力农业智能化，守护粮食安全 🌾</p>
  <p><i>Powered by PaddlePaddle & YOLOv10</i></p>
</div>
