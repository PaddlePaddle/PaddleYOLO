export FLAGS_allocator_strategy=auto_growth
name=n # 36.3 28.6 # rand24 36.6 27.4
name=s # # 44.0 34.7 # rand24 44.0 33.2
#name=m # 48.3 37.8 # rand24 48.6 36.6
#name=l # 50.8 39.5 # rand24 50.9 39.0
# name=x # 52.1 40.6 # rand24 51.9 40.1
model_type=yolov6_seg
job_name=yolov6_seg_${name}_300e_coco

# model_type=yolov6
# job_name=yolov6_${name}_300e_coco

config=configs/${model_type}/${job_name}.yml
log_dir=log_dir/${job_name}
weights=../weights/yolov6_seg_${name}_300e_coco.pdparams

# 1. training
#CUDA_VISIBLE_DEVICES=6 python tools/train.py -c ${config} -r ${weights} --amp
#python -m paddle.distributed.launch --log_dir=${log_dir} --gpus 0,1,2,3 tools/train.py -c ${config} --eval --amp

# 2. eval
#CUDA_VISIBLE_DEVICES=2 python tools/eval.py -c ${config} -o weights=https://paddledet.bj.bcebos.com/models/${job_name}.pdparams
CUDA_VISIBLE_DEVICES=7 python tools/eval.py -c ${config} -o weights=${weights} #--amp

#CUDA_VISIBLE_DEVICES=5 python tools/infer.py -c ${config} -o weights=${weights} --infer_img=demo/000000014439.jpg
#CUDA_VISIBLE_DEVICES=5 python tools/infer.py -c ${config} -o weights=${weights} --infer_img=demo/000000087038.jpg
#CUDA_VISIBLE_DEVICES=5 python tools/infer.py -c ${config} -o weights=${weights} --infer_img=demo/000000570688.jpg
#CUDA_VISIBLE_DEVICES=5 python tools/infer.py -c ${config} -o weights=${weights} --infer_img=demo/000000014439_640x640.jpg