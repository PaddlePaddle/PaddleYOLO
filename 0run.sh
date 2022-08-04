export FLAGS_allocator_strategy=auto_growth
name=l
model_type=yolov7
job_name=yolov7_${name}_300e_coco
config=configs/${model_type}/${job_name}.yml
log_dir=log_dir/${job_name}
weights=https://bj.bcebos.com/v1/paddledet/models/yolov7_${name}_300e_coco.pdparams
weights=../yolov7_tools/yolov7_${name}_300e_coco.pdparams

#python3.7 dygraph_print.py -c ${config} 2>&1 | tee yolov7_${name}_dy_print.txt

# 1. training
CUDA_VISIBLE_DEVICES=0 python3.7 tools/train.py -c ${config} --amp --eval #-r ${weights}
#python3.7 -m paddle.distributed.launch --log_dir=${log_dir} --gpus 0,1,2,3,4,5,6,7 tools/train.py -c ${config} --eval --amp

# 2. eval
#CUDA_VISIBLE_DEVICES=5 python3.7 tools/eval.py -c ${config} -o weights=${weights} #--classwise

# 3. tools infer
#CUDA_VISIBLE_DEVICES=1 python3.7 tools/infer.py -c ${config} -o weights=${weights} --infer_img=demo/000000014439_640x640.jpg

# 4.export model
#CUDA_VISIBLE_DEVICES=3 python3.7 tools/export_model.py -c ${config} -o weights=${weights} #exclude_nms=True trt=True

# 5. deploy infer
#CUDA_VISIBLE_DEVICES=7 python3.7 deploy/python/infer.py --model_dir=output_inference/${job_name} --image_file=demo/000000014439_640x640.jpg --device=GPU

# 6. deploy speed
#CUDA_VISIBLE_DEVICES=3 python3.7 deploy/python/infer.py --model_dir=output_inference/${job_name} --image_file=demo/000000014439_640x640.jpg --device=GPU --run_benchmark=True #--run_mode=trt_fp16

# 7. onnx speed
#paddle2onnx --model_dir output_inference/${job_name} --model_filename model.pdmodel --params_filename model.pdiparams --opset_version 12 --save_file ${job_name}.onnx
#/usr/local/TensorRT-8.0.3.4/bin/trtexec --onnx=${job_name}.onnx --workspace=4096 --avgRuns=10 --shapes=input:1x3x640x640 --fp16
