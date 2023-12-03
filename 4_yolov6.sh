export FLAGS_allocator_strategy=auto_growth
name=n # 37.5 # rand24 36.5
name=s # 44.8 # rand24 44.9
# name=m # 49.5 # rand24 48.7
# name=l # 52.2 # rand24 51.4

# model_type=yolov6_seg
# job_name=yolov6_seg_${name}_300e_coco

model_type=yolov6
job_name=yolov6_${name}_300e_coco

config=configs/${model_type}/${job_name}.yml
log_dir=log_dir/${job_name}
#weights=../weights/yolov6_seg_${name}_300e_coco.pdparams
weights=../weights/yolov6_${name}_300e_coco.pdparams

# 1. training
CUDA_VISIBLE_DEVICES=6 python tools/train.py -c ${config} -r ${weights} --amp
#python -m paddle.distributed.launch --log_dir=${log_dir} --gpus 0,1,2,3 tools/train.py -c ${config} --eval --amp

# 2. eval
#CUDA_VISIBLE_DEVICES=2 python tools/eval.py -c ${config} -o weights=https://paddledet.bj.bcebos.com/models/${job_name}.pdparams
#CUDA_VISIBLE_DEVICES=7 python tools/eval.py -c ${config} -o weights=${weights} #--amp

#CUDA_VISIBLE_DEVICES=5 python tools/infer.py -c ${config} -o weights=${weights} --infer_img=demo/000000014439.jpg
#CUDA_VISIBLE_DEVICES=5 python tools/infer.py -c ${config} -o weights=${weights} --infer_img=demo/000000087038.jpg
#CUDA_VISIBLE_DEVICES=5 python tools/infer.py -c ${config} -o weights=${weights} --infer_img=demo/000000570688.jpg
#CUDA_VISIBLE_DEVICES=5 python tools/infer.py -c ${config} -o weights=${weights} --infer_img=demo/000000014439_640x640.jpg