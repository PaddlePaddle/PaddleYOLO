# 水稻稻瘟病检测系统 - 启动配置示例
# Rice Blast Disease Detection System - Startup Configuration Example

# ================================
# 基本配置 (Basic Configuration)
# ================================

# 模型配置文件路径 (Model config file path)
# 修改为您的实际配置文件路径
CONFIG_PATH = "../configs/yolov10/rice_blast_yolov10_n.yml"

# 模型权重文件路径 (Model weights file path)
# 修改为您的实际权重文件路径
WEIGHTS_PATH = "../output/rice_blast_yolov10_n/model_final.pdparams"

# ================================
# 服务器配置 (Server Configuration)
# ================================

# 主机地址 (Host address)
# 127.0.0.1 - 只允许本地访问
# 0.0.0.0 - 允许局域网内其他设备访问
HOST = "127.0.0.1"

# 端口号 (Port number)
PORT = 5000

# ================================
# 模型配置 (Model Configuration)
# ================================

# 是否使用GPU (Use GPU)
# True - 使用GPU加速（需要安装paddlepaddle-gpu）
# False - 使用CPU（速度较慢）
USE_GPU = False

# ================================
# 检测配置 (Detection Configuration)
# ================================

# 默认置信度阈值 (Default confidence threshold)
# 范围：0.0 - 1.0
# 推荐值：0.5
DEFAULT_THRESHOLD = 0.5

# ================================
# 文件配置 (File Configuration)
# ================================

# 上传文件大小限制 (Upload file size limit)
# 单位：MB
MAX_FILE_SIZE_MB = 16

# 上传文件保存目录 (Upload directory)
UPLOAD_FOLDER = "uploads"

# ================================
# 示例使用 (Example Usage)
# ================================

# Windows命令行启动：
# python app.py --config ../configs/yolov10/rice_blast_yolov10_n.yml --weights ../output/rice_blast_yolov10_n/model_final.pdparams

# 使用GPU启动：
# python app.py --config ../configs/yolov10/rice_blast_yolov10_n.yml --weights ../output/rice_blast_yolov10_n/model_final.pdparams --use_gpu

# 指定端口启动：
# python app.py --config ../configs/yolov10/rice_blast_yolov10_n.yml --weights ../output/rice_blast_yolov10_n/model_final.pdparams --port 8080

# 使用批处理脚本启动（推荐）：
# run_app.bat

# ================================
# 常见问题 (FAQ)
# ================================

# Q: 如何知道我的模型文件路径？
# A: 查看output文件夹，找到您训练的模型目录，里面有model_final.pdparams文件

# Q: 如何使用GPU？
# A: 首先安装paddlepaddle-gpu版本，然后启动时添加 --use_gpu 参数

# Q: 为什么检测速度很慢？
# A: 可能是使用了CPU模式，建议使用GPU加速

# Q: 如何访问Web界面？
# A: 启动后在浏览器中打开 http://127.0.0.1:5000

# Q: 如何让其他设备访问？
# A: 启动时使用 --host 0.0.0.0 参数，然后在其他设备上访问本机IP
