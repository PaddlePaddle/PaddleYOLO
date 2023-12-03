export FLAGS_allocator_strategy=auto_growth
name=s # # 22.3 # rand24 22.8
#name=m # 24.8 # rand24 26.2
#name=l # 27.6 # rand24 27.2
model_type=yolov6
job_name=yolov6lite_${name}_400e_coco

config=configs/${model_type}/yolov6lite/${job_name}.yml
log_dir=log_dir/${job_name}
weights=../weights/yolov6lite_${name}_400e_coco.pdparams
#weights=https://paddledet.bj.bcebos.com/models/${job_name}.pdparams

# 1. training
CUDA_VISIBLE_DEVICES=6 python tools/train.py -c ${config} -r ${weights} #--amp
#python -m paddle.distributed.launch --log_dir=${log_dir} --gpus 0,1,2,3,4,5,6,7 tools/train.py -c ${config} --eval --amp

# 2. eval
#CUDA_VISIBLE_DEVICES=2 python tools/eval.py -c ${config} -o weights=https://paddledet.bj.bcebos.com/models/${job_name}.pdparams
#CUDA_VISIBLE_DEVICES=7 python tools/eval.py -c ${config} -o weights=${weights} #--amp

#CUDA_VISIBLE_DEVICES=5 python tools/infer.py -c ${config} -o weights=${weights} --infer_img=demo/000000014439.jpg
#CUDA_VISIBLE_DEVICES=5 python tools/infer.py -c ${config} -o weights=${weights} --infer_img=demo/000000087038.jpg
#CUDA_VISIBLE_DEVICES=5 python tools/infer.py -c ${config} -o weights=${weights} --infer_img=demo/000000570688.jpg
#CUDA_VISIBLE_DEVICES=5 python tools/infer.py -c ${config} -o weights=${weights} --infer_img=demo/000000014439_640x640.jpg