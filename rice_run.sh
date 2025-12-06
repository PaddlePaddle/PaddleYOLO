model_name=yolov10 # 可修改，如 ppyoloe
job_name=rice_blast_yolov10_n # 可修改，如 ppyoloe_plus_crn_s_80e_coco

config=configs/${model_name}/${job_name}.yml
log_dir=log_dir/${job_name}
weights=output/${job_name}/model_final.pdparams

# 1.训练（单卡/多卡），加 --eval 表示边训边评估，加 --amp 表示混合精度训练
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/yolov8/yolov8_n_100e_64b8b_rice80.yml --eval --amp
# python -m paddle.distributed.launch --log_dir=${log_dir} --gpus 0,1,2,3,4,5,6,7 tools/train.py -c ${config} --eval --amp

# 2.评估，加 --classwise 表示输出每一类mAP
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c ${config} -o weights=${weights} --classwise

# 3.预测 (单张图/图片文件夹）
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/yolov8/yolov8_n_100e_64b8b_rice200.yml -o weights=output/yolov8_n_100e_64b8b_rice200/best_model.pdparams --infer_dir=H:\\dataset\\rice\\80\\images --draw_threshold=0.5 --output_dir=output/yolov8_n_100e_64b8b_rice200/inference_results
# CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c ${config} -o weights=${weights} --infer_dir=demo/ --draw_threshold=0.5

# 4.导出模型，以下3种模式选一种
## 普通导出，加trt表示用于trt加速，对NMS和silu激活函数提速明显
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c ${config} -o weights=${weights} # trt=True

## exclude_post_process去除后处理导出，返回和YOLOv5导出ONNX时相同格式的concat后的1个Tensor，是未缩放回原图的坐标+分类置信度
# CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c ${config} -o weights=${weights} exclude_post_process=True # trt=True

# 5.部署预测，注意不能使用去除后处理导出后的模型去预测
CUDA_VISIBLE_DEVICES=0 python deploy/python/infer.py --model_dir=output_inference/${job_name} --image_file=demo/000000014439_640x640.jpg --device=GPU

# 6.部署测速，加 “--run_mode=trt_fp16” 表示在TensorRT FP16模式下测速，注意如需用到 trt_fp16 则必须为加 trt=True 导出的模型
CUDA_VISIBLE_DEVICES=0 python deploy/python/infer.py --model_dir=output_inference/${job_name} --image_file=demo/000000014439_640x640.jpg --device=GPU --run_benchmark=True # --run_mode=trt_fp16

# 7.onnx导出，一般结合 exclude_post_process去除后处理导出的模型
paddle2onnx --model_dir output_inference/${job_name} --model_filename model.pdmodel --params_filename model.pdiparams --opset_version 12 --save_file ${job_name}.onnx

# 8.onnx trt测速
trtexec --onnx=${job_name}.onnx --workspace=4096 --avgRuns=10 --shapes=input:1x3x640x640 --fp16
trtexec --onnx=${job_name}.onnx --workspace=4096 --avgRuns=10 --shapes=input:1x3x640x640 --fp32