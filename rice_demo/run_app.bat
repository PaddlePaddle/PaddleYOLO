@echo off
REM Rice Blast Disease Detection System Startup Script
REM Usage: run_app.bat [config_file] [weights_file]

echo ================================================
echo   Rice Blast Disease Detection System
echo   水稻稻瘟病智能检测系统
echo ================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH!
    echo 错误：Python未安装或不在环境变量中！
    pause
    exit /b 1
)

REM Set default config and weights if not provided
set CONFIG=%1
set WEIGHTS=%2

if "%CONFIG%"=="" (
    set CONFIG=..\configs\yolov10\rice_blast_yolov10_n.yml
    echo Using default config: %CONFIG%
)

if "%WEIGHTS%"=="" (
    set WEIGHTS=..\output\rice_blast_yolov10_n\model_final.pdparams
    echo Using default weights: %WEIGHTS%
)

REM Check if files exist
if not exist "%CONFIG%" (
    echo Error: Config file not found: %CONFIG%
    echo 错误：配置文件未找到！
    pause
    exit /b 1
)

if not exist "%WEIGHTS%" (
    echo Error: Weights file not found: %WEIGHTS%
    echo 错误：权重文件未找到！
    pause
    exit /b 1
)

echo.
echo Starting web application...
echo 正在启动Web应用...
echo.
echo Config: %CONFIG%
echo Weights: %WEIGHTS%
echo.
echo The application will open at: http://127.0.0.1:5000
echo 应用将在以下地址打开: http://127.0.0.1:5000
echo.
echo Press Ctrl+C to stop the server
echo 按 Ctrl+C 停止服务器
echo.

REM Start the Flask application
python app.py --config %CONFIG% --weights %WEIGHTS% --port 5000

pause
