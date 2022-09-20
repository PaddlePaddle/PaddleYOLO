import copy
import numpy as np
import paddle
from ppdet.modeling.necks.yolo_pafpn import YOLOPAFPN

paddle.seed(33)
np.random.seed(33)

def randtool(dtype, low, high, shape):
    """
    np random tools
    """
    if dtype == "int":
        return np.random.randint(low, high, shape)


    elif dtype == "float":
        return low + (high - low) * np.random.random(shape)

def train():
    paddle.seed(33)
    np.random.seed(33)
    paddle.set_default_dtype("float64")
    input = [paddle.to_tensor(randtool("float", -1, 1, shape=[1, 256, 256, 256]).astype("float64")),
             paddle.to_tensor(randtool("float", -1, 1, shape=[1, 512, 128, 128]).astype("float64")),
             paddle.to_tensor(randtool("float", -1, 1, shape=[1, 1024, 64, 64]).astype("float64"))]
    net = YOLOPAFPN(in_channels=[256, 512, 1024],use_att='ASFF')

    print(net)
    # print("net parameters is: ", net.parameters())


    # opt = paddle.optimizer.SGD(learning_rate=0.00001, parameters=net.parameters())
    # dygraph train
    loss = net(input)
    return loss


dy_out_final = train()

# st_out_final = train(True)

# # 结果打印

print("dy_out_final", dy_out_final)

# print("st_out_final", st_out_final)

# print(np.array_equal(dy_out_final.numpy(), st_out_final.numpy()))

#

# print("diff is: ", dy_out_final - st_out_final)
