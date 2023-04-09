export FLAGS_allocator_strategy=auto_growth
name=n
model_type=yolov5
job_name=yolov5_${name}_300e_coco
job_name=yolov5_ins_${name}_300e_coco
config=configs/${model_type}/${job_name}.yml
log_dir=log_dir/${job_name}
weights=/paddle/mypaddleyolo/yolov5/yolov5_ins_${name}_300e_coco.pdparams

# 1. training
#CUDA_VISIBLE_DEVICES=3 python3.7 tools/train.py -c ${config} --amp #-r ${weights}
#python3.7 -m paddle.distributed.launch --log_dir=${log_dir} --gpus 1,2,3,4,5,6 tools/train.py -c ${config} --eval # --amp

# 2. eval
#CUDA_VISIBLE_DEVICES=0 python3.7 tools/eval.py -c ${config} -o weights=https://paddledet.bj.bcebos.com/models/${job_name}.pdparams
CUDA_VISIBLE_DEVICES=3 python3.7 tools/eval.py -c ${config} -o weights=${weights} #--amp

# 3. tools infer
#CUDA_VISIBLE_DEVICES=7 python3.7 tools/infer.py -c ${config} -o weights=${weights} --infer_img=demo/000000014439_640x640.jpg
#CUDA_VISIBLE_DEVICES=6 python3.7 tools/infer.py -c ${config} -o weights=${weights} --infer_img=demo/000000329219.jpg


# 4.导出模型
#CUDA_VISIBLE_DEVICES=1 python3.7 tools/export_model.py -c ${config} -o weights=${weights} #exclude_nms=True trt=True

# 5.部署预测
#CUDA_VISIBLE_DEVICES=1 python3.7 deploy/python/infer.py --model_dir=output_inference/${job_name} --image_file=demo/000000014439_640x640.jpg --device=GPU

# 6.部署测速
#CUDA_VISIBLE_DEVICES=1 python3.7 deploy/python/infer.py --model_dir=output_inference/${job_name} --image_file=demo/000000014439_640x640.jpg --device=GPU --run_benchmark=True # --run_mode=trt_fp16

# 7.onnx导出
#paddle2onnx --model_dir output_inference/${job_name} --model_filename model.pdmodel --params_filename model.pdiparams --opset_version 12 --save_file ${job_name}.onnx
