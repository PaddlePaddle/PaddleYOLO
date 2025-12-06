# 🔧 Bug 修复记录

## 问题描述

### 问题 1：错误的导入路径
启动应用时出现错误：
```
[12/06 12:51:33] rice_blast_app ERROR: Prediction error: No module named 'ppdet.utils.coco_utils'
```

### 问题 3：数据类型错误
```
TypeError: object of type 'numpy.float32' has no len()
```
在 `app.py` 第 212 行：`if len(bbox) >= 6:`

## 问题原因

### 原因 1
在 `app.py` 的 `predict_image` 函数中使用了错误的导入路径：
```python
from ppdet.utils.coco_utils import get_categories  # ❌ 错误
```

正确的导入路径应该是：
```python
from ppdet.data.source.category import get_categories  # ✅ 正确
```

### 原因 2
代码中导入了不存在的模块：
```python
from ppdet.data.reader import create_reader  # ❌ 不存在
```
这个导入是不必要的，应该删除。

### 原因 3
错误地使用了数组索引：
```python
bbox_data = outs['bbox'][0]  # ❌ 错误，多了一层索引
bbox = bbox_data[i]
if len(bbox) >= 6:  # bbox 实际是 float32，不是数组
```

正确的做法：
```python
bbox_data = outs['bbox']  # ✅ 直接使用，shape 是 [N, 6]
bbox = bbox_data[i]  # 现在 bbox 是一个长度为6的数组
if len(bbox) >= 6:  # ✅ 正确
```

## 修复内容

### 1. 修复导入路径
将所有的 `ppdet.utils.coco_utils` 改为 `ppdet.data.source.category`

### 2. 添加缺失的 cv2 导入
在文件开头添加：
```python
import cv2
```

### 3. 移除不存在的导入
删除以下不必要的导入：
```python
from ppdet.data.reader import create_reader  # 已删除
from ppdet.core.workspace import create  # 已删除
from ppdet.data.transform.operators import DecodeImage, Resize, NormalizeImage, Permute  # 已删除
```

### 4. 简化推理逻辑
直接使用模型进行推理，而不是调用子进程，使代码更加稳定可靠。

### 5. 修复 bbox 解析
正确处理 numpy 数组的维度和索引：
```python
# 移除错误的 [0] 索引
bbox_data = outs['bbox']  # 而不是 outs['bbox'][0]

# 添加形状检查
if len(bbox_data.shape) == 2:
    bbox = bbox_data[i]  # 正确获取每个检测框

# 显式类型转换
class_id = float(bbox[0])
score = float(bbox[1])
x1, y1, x2, y2 = float(bbox[2]), float(bbox[3]), float(bbox[4]), float(bbox[5])
```

## 测试方法

### 快速测试
```cmd
cd rice_demo
python app.py --config ../configs/yolov8/yolov8_n_100e_16b8b_rice200.yml --weights ../output/yolov8_n_100e_16b8b_rice/best_model.pdparams
```

### 完整测试步骤

1. **启动应用**
```cmd
cd e:\projects\myPaddleYOLO\rice_demo
python app.py --config ../configs/yolov8/yolov8_n_100e_16b8b_rice200.yml --weights ../output/yolov8_n_100e_16b8b_rice/best_model.pdparams
```

2. **打开浏览器**
访问 http://127.0.0.1:5000

3. **测试上传功能**
- 上传单张图片
- 调整置信度阈值
- 点击"开始检测"
- 查看检测结果

4. **测试文件夹功能**
- 输入文件夹路径
- 点击"加载"按钮
- 使用前后翻页
- 对多张图片进行检测

## 已修复的文件

- ✅ `rice_demo/app.py` - 修复导入路径和添加 cv2

## 预期结果

应用应该能够正常启动并运行推理，不再出现 `No module named` 错误。

## 如果仍有问题

### 检查清单

1. **确认 PaddleDetection 安装正确**
```cmd
python -c "from ppdet.data.source.category import get_categories; print('OK')"
```

2. **确认 OpenCV 已安装**
```cmd
python -c "import cv2; print(cv2.__version__)"
```
如果没有安装：
```cmd
pip install opencv-python
```

3. **确认配置文件路径正确**
```cmd
dir ..\configs\yolov8\yolov8_n_100e_16b8b_rice200.yml
```

4. **确认权重文件路径正确**
```cmd
dir ..\output\yolov8_n_100e_16b8b_rice\best_model.pdparams
```

### 查看详细错误信息

如果还有错误，查看控制台输出的完整 traceback 信息，这将帮助定位问题。

## 更新日志

### v1.0.3 (2025-12-06 13:00)
- 🐛 修复: bbox 数据结构解析错误
- 📝 添加: bbox 形状检查和调试日志
- ♻️ 优化: 改进 numpy 数组索引处理

### v1.0.2 (2025-12-06 12:57)
- 🐛 修复: 移除不存在的 `create_reader` 导入
- ♻️ 优化: 简化推理代码，移除不必要的导入

### v1.0.1 (2025-12-06 12:52)
- 🐛 修复: 错误的 get_categories 导入路径
- ➕ 添加: cv2 (OpenCV) 导入
- ♻️ 重构: 简化推理逻辑，提高稳定性
- 📝 文档: 添加故障排查指南

---

**修复完成！** 现在应该可以正常运行了。如有其他问题，请查看控制台输出的错误信息。
