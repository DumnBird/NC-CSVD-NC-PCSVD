import tensorflow  as tf
import os
from tensorflow.python import pywrap_tensorflow
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES']='2'


reader = pywrap_tensorflow.NewCheckpointReader('/home/test01/csw/resnet34/ckpt3_20200617/resnet-34-accTensor("accuracy:0", shape=(), dtype=float32)-5')  # tf.train.NewCheckpointReader
var_to_shape_map = reader.get_variable_to_shape_map()

new_var_dic = {}

for key in var_to_shape_map:
    value = reader.get_tensor(key)

    new_var_dic[key] = value

np.save('71.09parameter.npy', new_var_dic)