import numpy as np
import os
import tensorflow as tf
from numpy import *
from tensorflow.python import pywrap_tensorflow

R_c= [0.0625, 0.125, 0.1875, 0.25, 0.5]


R_i = [[0.11498091603053436, 0.1049618320610687, 0.09494274809160305, 0.08492366412213741, 0.04484732824427481],
        [0.17748091603053434, 0.1674618320610687, 0.15744274809160305, 0.1474236641221374, 0.10734732824427481],
        [0.23998091603053434, 0.2299618320610687, 0.21994274809160305, 0.2099236641221374, 0.16984732824427481],
         [0.36498091603053434, 0.3549618320610687, 0.3449427480916031, 0.3349236641221374, 0.2948473282442748],
         [0.48998091603053434, 0.4799618320610687, 0.4699427480916031, 0.4599236641221374, 0.4198473282442748],
          [0.7399809160305344, 0.7299618320610687, 0.7199427480916031, 0.7099236641221374, 0.6698473282442748]]


def svd_compression(weight, rank_rate):
    F1, F2, I, O = weight.shape  # weight[F1,F2,I,O]
    #以o为基准计算的rank
    rank = int(O*rank_rate)

    W = np.reshape(np.transpose(weight,(0,2,1,3)), [F1*I, F2*O])
    U_2d, S, V_2d = np.linalg.svd(W, full_matrices=True)

    U_2d  = np.dot(U_2d[:, 0:rank].copy(),np.diag(S)[0:rank, 0:rank].copy())
    U_4d = np.transpose(np.reshape(U_2d, [F1, I, 1, rank]),(0,2,1,3))

    V_2d = V_2d[0:rank, :].copy()
    V_4d = np.transpose(np.reshape(V_2d,[1,rank,F2,O]),(0,2,1,3))

    return U_4d, V_4d

def svd_compression_restore(U_4d, V_4d):

    h, _, i, r1 = U_4d.shape
    assert(_==1)
    _, w, r2, o = V_4d.shape
    assert(r1==r2)
    assert(_==1)

    #[3,1,i,r]
    U_2d = np.reshape(np.transpose(U_4d, [0,2,1,3]), [h*i, r1])
    V_2d = np.reshape(np.transpose(V_4d, [0,2,1,3]), [r2, w*o])

    W_2d = np.dot(U_2d, V_2d)
    W_4d = np.transpose(np.reshape(W_2d, [h,i,w,o]), [0,2,1,3])

    return W_4d






#with W_orig， W_i, rc, W_4d_share, get U_4d_partly_shared,V_4d_shared
def iteration_pshare(para_dict, Wi_4d_dict, r):
    W_4d_share_dict = {}
    UV_4d_share_dict = {}

    repeat_num = [4,6,3]

    for layer in range(2, 5):
        for repeat in range(repeat_num[layer-2]):
            for conv in ['1','2']:

                #64q 128q不分解
                if repeat==0 and conv=='1':
                    pass
                else:
  
                    W_list=[]

                    for repeat in range(repeat_num[layer-2]):
                        if repeat==0 and conv=='1':
                            pass
                        else:
                            W_orig_name = 'layer' + str(layer) + '.' + str(repeat) + '.conv' + conv + '.weight'
                            weight_name = 'layer' + str(layer) + '.' + str(repeat) + '.conv' + conv + '.'

                            W_orig = para_dict[W_orig_name]


                            W_diff_4d = W_orig - Wi_4d_dict[weight_name+'UV_restore']
                            W_list.append(W_diff_4d)

                    W_diff = np.mean(W_list, axis=0)
                    U_4d, V_4d = svd_compression(W_diff, r)

                    weight_share_name = 'layer' + str(layer) + '.conv' + conv +'.'
                    UV_4d_share_dict[weight_share_name + 'U_shared'] = U_4d
                    UV_4d_share_dict[weight_share_name + 'V_shared'] = V_4d

                    W_4d_share_dict[weight_share_name + 'W_4d_share'] = svd_compression_restore(U_4d, V_4d)

    return UV_4d_share_dict, W_4d_share_dict


#initialize independent components
def initialize_independent(para_dict, ri_rate):
    Wi_4d_dict = {}
    UV_4d_indep_dict = {}

    repeat_num = [4,6,3]

    for layer in range(2, 5):
        for repeat in range(repeat_num[layer-2]):
            for conv in ['1','2']:

                #64q 128q不分解
                if repeat==0 and conv=='1':
                    pass
                else:
                    W_orig_name = 'layer' + str(layer) + '.' + str(repeat) + '.conv' + conv + '.weight'
                    indep_weight_name = 'layer' + str(layer) + '.' + str(repeat) + '.conv' + conv + '.'

                    U_4d, V_4d = svd_compression(para_dict[W_orig_name], ri_rate)

                    UV_4d_indep_dict[indep_weight_name+'U'] = U_4d
                    UV_4d_indep_dict[indep_weight_name+'V'] = V_4d

                    Wi_4d_dict[indep_weight_name+'UV_restore'] = svd_compression_restore(U_4d, V_4d)
    return UV_4d_indep_dict, Wi_4d_dict



def iteration_independent(para_dict, W_4d_share_dict, ri_rate):
    Wi_4d_dict = {}
    UV_4d_indep_dict = {}

    repeat_num = [4,6,3]

    for layer in range(2, 5):
        for repeat in range(repeat_num[layer-2]):
            for conv in ['1','2']:
                #64q 128q不分解
                if repeat==0 and conv=='1':
                    pass
                else:
                    orig_weight_name = 'layer' + str(layer) + '.' + str(repeat) + '.conv' + conv + '.weight'
                    share_weight_name = 'layer' + str(layer) + '.conv' + conv +  '.W_4d_share'
                    indep_weight_name = 'layer' + str(layer) + '.' + str(repeat) + '.conv' + conv + '.'

                    U_4d, V_4d = svd_compression( para_dict[orig_weight_name]- W_4d_share_dict[share_weight_name], ri_rate)
                    UV_4d_indep_dict[indep_weight_name+'U'] = U_4d
                    UV_4d_indep_dict[indep_weight_name+'V'] = V_4d

                    Wi_4d_dict[indep_weight_name+'UV_restore'] = svd_compression_restore(U_4d, V_4d)

    return UV_4d_indep_dict, Wi_4d_dict


reader = pywrap_tensorflow.NewCheckpointReader('/home/test01/csw/resnet34/ckpt3_20200617/resnet-34-accTensor("accuracy:0", shape=(), dtype=float32)-5')  # tf.train.NewCheckpointReader
var_to_shape_map = reader.get_variable_to_shape_map()
var_dict = {}
for key in var_to_shape_map:
    value = reader.get_tensor(key)
    var_dict[key] = value

path = './resnet34_csvd_npy'
os.makedirs(path,exist_ok=True)




for ci in range(len(R_i)):
    for cc in range(len(R_c)):
        rc_rate = R_c[cc]
        ri_rate = R_i[ci][cc]

        if ri_rate<0:
            pass
        else:
            # print(r_c, ' ', r_i)
            UV_4d_indep_dict, Wi_4d_dict = initialize_independent(var_dict, ri_rate)
            UV_4d_share_dict, W_4d_share_dict  = iteration_pshare(var_dict, Wi_4d_dict, rc_rate)

            proposed_dict = {**UV_4d_indep_dict, **UV_4d_share_dict}
            np.save(path+'/iter0_npy_rc=%f_ri=%f.npy'%(round(rc_rate,4),round(ri_rate,6)), proposed_dict)
            for item in proposed_dict:
                print(item)
                print(proposed_dict[item].shape)

            print("start iteration")
            for i in range(10):
                print(i)
                UV_4d_indep_dict, Wi_4d_dict = iteration_independent(var_dict, W_4d_share_dict, ri_rate)

                UV_4d_share_dict, W_4d_share_dict = iteration_pshare(var_dict, Wi_4d_dict, rc_rate)

                if i==9:
                    print("%dth iteration of r1=%8f,c2=%8f has finished!!" % (i, rc_rate, ri_rate))

                if i==3 or i==6 or i==9:
                    proposed_dict = {**UV_4d_indep_dict, **UV_4d_share_dict}
                    np.save(path+'/iter%d_npy_rc=%f_ri=%f.npy'%(i,round(rc_rate,4),round(ri_rate,6)), proposed_dict)







