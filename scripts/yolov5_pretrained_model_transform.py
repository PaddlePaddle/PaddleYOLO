
import paddle

model = paddle.load("./pretrained_models/yolov5_l_300e_coco.pdparams")
new_model = {}

replace_map = {
    'conv.weight': 'resnet_unit.filter_x',
    'bn.weight': 'resnet_unit.scale_x',
    'bn.bias': 'resnet_unit.bias_x',
    'bn._mean': 'resnet_unit.mean_x',
    'bn._variance': 'resnet_unit.var_x',
}

for k,v in model.items():
    val = k.split('.', 2)[-1]
    new_key = k
    if val in replace_map.keys():
        new_key = k.replace(val, replace_map[val])
    val = k.split('.', 3)[-1]
    if val in replace_map.keys():
        new_key = k.replace(val, replace_map[val])
    val = k.split('.', 4)[-1]
    if val in replace_map.keys():
        new_key = k.replace(val, replace_map[val])
    val = k.split('.', 5)[-1]
    if val in replace_map.keys():
        new_key = k.replace(val, replace_map[val])
    val = k.split('.', 6)[-1]
    if val in replace_map.keys():
        new_key = k.replace(val, replace_map[val])
    val = k.split('.', 7)[-1]
    if val in replace_map.keys():
        new_key = k.replace(val, replace_map[val])
    # print("val = ", val, "  k = ", k, " ==> new_k = ", new_key)
    print(new_key)
    new_model[new_key] = v

paddle.save(new_model, "./pretrained_models/yolov5_l_300e_coco_fusion.pdparams")

