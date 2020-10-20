import numpy as np
import os
import tensorflow as tf
from numpy import *


R_c= [0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5]


R_i = [ [0.020089285714285716, 0.008928571428571428, -0.013392857142857142, -0.05803571428571429, -0.14732142857142858, -0.32589285714285715],
        [0.05133928571428571, 0.04017857142857143, 0.017857142857142856, -0.026785714285714284, -0.11607142857142858, -0.29464285714285715],
        [0.11383928571428571, 0.10267857142857142, 0.08035714285714286, 0.03571428571428571, -0.05357142857142857, -0.23214285714285715],
        [0.23883928571428573, 0.22767857142857142, 0.20535714285714285, 0.16071428571428573, 0.07142857142857142, -0.10714285714285714],
        [0.3638392857142857, 0.35267857142857145, 0.33035714285714285, 0.2857142857142857, 0.19642857142857142, 0.017857142857142856],
        [0.4888392857142857, 0.47767857142857145, 0.45535714285714285, 0.4107142857142857, 0.32142857142857145, 0.14285714285714285],
        [0.7388392857142857, 0.7276785714285714, 0.7053571428571429, 0.6607142857142857, 0.5714285714285714, 0.39285714285714285]]



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





def iteration_pshare(para_dict, Wi_4d_dict, r):
    W_4d_share_dict = {}
    UV_4d_share_dict = {}

    for layer in range(2,5):
        W_list=[]

        h, w, i, o = para_dict['layer' + str(layer) + '.0' + '.conv1' + '.weight'].shape

        for repeat in range(2):
            W_orig_name = 'layer' + str(layer) + '.' + str(repeat) + '.conv1' + '.weight'
            weight_name = 'layer' + str(layer) + '.' + str(repeat) + '.conv1' + '.'

            W_orig = para_dict[W_orig_name]


            W_diff_2d = np.reshape( np.transpose(W_orig-Wi_4d_dict[weight_name+'UV_restore'], [0,2,1,3]), [h*W_orig.shape[2], w*o])
            W_list.append(W_diff_2d)

        W = np.vstack([W_list[0],W_list[1]])
        u1,s1,v1= np.linalg.svd(W, full_matrices=True)


        U_2d_partly_shared = np.dot(u1[:, 0:int(o*r)].copy(), np.diag(s1)[0:int(o*r), 0:int(o*r)].copy())

        U_2d_partly_shared1 = U_2d_partly_shared[0:h*i,:].copy()

        U_4d_partly_shared1 = np.transpose(np.reshape(U_2d_partly_shared1, [h, i, 1, int(o*r)]), [0,2,1,3])

        U_2d_partly_shared2 = U_2d_partly_shared[h*i:,:].copy()

        U_4d_partly_shared2 = np.transpose(np.reshape(U_2d_partly_shared2, [h, o, 1, int(o*r)]), [0,2,1,3])
        for repeat in range(2):
            name_weight = 'layer' + str(layer) + '.' + str(repeat) + '.conv1' + '.U_partly_shared'
            UV_4d_share_dict[name_weight] = locals()['U_4d_partly_shared' + str(repeat+1)]


        V_2d_shared = v1[0:int(o*r), :].copy()

        V_4d_shared = np.transpose(np.reshape(V_2d_shared, [1, int(o*r), w, o]), (0,2,1,3))
        UV_4d_share_dict['layer' + str(layer) + '.conv1' + '.V_shared'] = V_4d_shared

        W1_4d_share_restore = svd_compression_restore(U_4d_partly_shared1, V_4d_shared)
        W2_4d_share_restore = svd_compression_restore(U_4d_partly_shared2, V_4d_shared)
        for repeat in range(2):
            name_weight1 = 'layer' + str(layer) + '.' + str(repeat) + '.conv1' +  '.W_4d_share'
            W_4d_share_dict[name_weight1] = locals()['W'+str(repeat+1)+'_4d_share_restore']

    return UV_4d_share_dict, W_4d_share_dict



def initialize_independent(para_dict, ri_rate):
    Wi_4d_dict = {}
    UV_4d_indep_dict = {}
    for layer in range(2, 5):
        for reapeat in range(2):
            W_orig_name = 'layer' + str(layer) + '.' + str(reapeat) + '.conv1' + '.weight'
            indep_weight_name = 'layer' + str(layer) + '.' + str(reapeat) + '.conv1' + '.'

            U_4d, V_4d = svd_compression(para_dict[W_orig_name], ri_rate)

            UV_4d_indep_dict[indep_weight_name+'U'] = U_4d
            UV_4d_indep_dict[indep_weight_name+'V'] = V_4d

            Wi_4d_dict[indep_weight_name+'UV_restore'] = svd_compression_restore(U_4d, V_4d)
    return UV_4d_indep_dict, Wi_4d_dict



def iteration_independent(para_dict, W_4d_share_dict, ri_rate):
    Wi_4d_dict = {}
    UV_4d_indep_dict = {}
    for layer in range(2,5):
        for repeat in range(2):
            orig_weight_name = 'layer' + str(layer) + '.' + str(repeat) + '.conv1' + '.weight'
            share_weight_name = 'layer' + str(layer) + '.' + str(repeat) + '.conv1' +  '.W_4d_share'
            indep_weight_name = 'layer' + str(layer) + '.' + str(repeat) + '.conv1' + '.'

            U_4d, V_4d = svd_compression( para_dict[orig_weight_name]- W_4d_share_dict[share_weight_name], ri_rate)
            UV_4d_indep_dict[indep_weight_name+'U'] = U_4d
            UV_4d_indep_dict[indep_weight_name+'V'] = V_4d

            Wi_4d_dict[indep_weight_name+'UV_restore'] = svd_compression_restore(U_4d, V_4d)
    return UV_4d_indep_dict, Wi_4d_dict


var_dict = np.load('/home/test01/csw/resnet8/tt/parameter.npy', allow_pickle=True).item()
path = './pcsvd_npy'
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
            for i in range(16):
                print(i)
                UV_4d_indep_dict, Wi_4d_dict = iteration_independent(var_dict, W_4d_share_dict, ri_rate)

                UV_4d_share_dict, W_4d_share_dict = iteration_pshare(var_dict, Wi_4d_dict, rc_rate)

                if i==9:
                    print("%dth iteration of r1=%8f,c2=%8f has finished!!" % (i, rc_rate, ri_rate))

                if i==3 or i==6 or i==9 or i==15:
                    proposed_dict = {**UV_4d_indep_dict, **UV_4d_share_dict}
                    np.save(path+'/iter%d_npy_rc=%f_ri=%f.npy'%(i,round(rc_rate,4),round(ri_rate,6)), proposed_dict)






