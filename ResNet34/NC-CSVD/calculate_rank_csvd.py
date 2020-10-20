'''
resenet34 csvd  64layer不分解  64q*128q用svd分解
对标resnet34 svd 压缩率 

给定resnet34svd rank， rc, 求ri， 与对应的压缩率
'''

import numpy as  np 




#resnet8
# in_out = [[64,128],[128,128],[128,256],[256,256],[256,512],[512,512]]


# Rank_rate_svd_str = ['1/32','1/16','1/8','1/4', '3/8', '1/2', '3/4']
Rank_rate_svd_str = ['1/8','3/16','1/4', '3/8', '1/2', '3/4']
Rank_rate_svd_str = ['3/16','1/4', '3/8', '1/2', '3/4']

Rank_rate_svd = [eval(i) for i in Rank_rate_svd_str]


# Rc_rate_str = ['1/64','1/32','1/16','1/8', '1/4','1/2']
Rc_rate_str = ['1/16','1/8','3/16', '1/4','1/2']
Rc_rate = [eval(i) for i in Rc_rate_str]




#paramter in shorcut downsample layer  and the 3*3*3*64layer
def other_parameter():
    #other layer + fc layer + shortcut downsample layer
    num = 7*7*64*3 + 3*3*64*64*6 + 512*1000 +1000 + 64*128 + 128*256 + 256*512
    return num

#128*128  256*256  512*512 layer
def orig():
    num =  3*3*128*128*7 + 3*3*256*256*11 + 3*3*512*512*5 +\
         + 3*3*64*128    + 3*3*128*256    + 3*3*256*512
    return num 

print('orig num of parameters')
print(other_parameter()+orig())


#num of parameter in the svd model under rank_rate_svd
def num_svd(rank_rate_svd):
    #svd以o 为基准计算rank 
    o = [128, 256, 512] #output size
    repeat_num = [4,6,3]
    Rank_svd = [int(item*rank_rate_svd) for item in o]


    num = 0
    for i in range(len(o)):
        #filter = 3*3； svd
        #64q*128q + 128q*128q
        num += 3*o[i] * Rank_svd[i] + 3*(o[i]/2) * Rank_svd[i] +\
               3*o[i]*Rank_svd[i]*2 * (2*repeat_num[i]-1)
               
    return num






#With rank_rate_svd, rc_rate, calculate ri_rate
def count_csvd_ri(rank_rate_svd, rc_rate, print_ph=0):
    #Resnet34 64q*28q采用svd  128q*128q采用csvd

    o = [128, 256, 512] #output size
    repeat_num = [4,6,3]

    num_share_para = 0
    #参照svd，以o为基准
    rank_c  = [int(item*rc_rate) for item in o]
    for i in range(len(o)):
        num_share_para += 3*o[i]*rank_c[i]*2

    num_svd_para = 0
    Rank_svd = [int(item*rank_rate_svd) for item in o]
    for i in range(len(o)):
        num_svd_para += 3*o[i]*Rank_svd[i] + 3*(o[i]/2)*Rank_svd[i]

    num_indep_para = num_svd(rank_rate_svd) - num_share_para - num_svd_para

    #num_indep_para = sum[(3*O + 3*O) * (O*ri_rate) * (2*repeat[i]-1)] = sum[6*O*O * (2*repeat[i]-1)]*ri_rate)
    coefficient = 0 
    for i in range(len(o)):
        coefficient += 6*o[i]*o[i] * (2*repeat_num[i]-1)
    ri_rate = num_indep_para / coefficient

    if print_ph:
        print(ri_rate)

    return ri_rate


def num_csvd(rank_rate_svd, rc_rate, ri_rate, print_ph=False):
    #求csvd resnet34 分解层总参数量
    if ri_rate<0:
        return 0

    o = [128, 256, 512] #output size
    repeat_num = [4,6,3]
    #参照svd，以o为基准
    #可能会给了（64q,128q）层更多share信息？如果效果不行，就调整算法，小的以i计算
    rank_c  = [int(item*rc_rate) for item in o]

    num_share_para = 0
    for i in range(len(o)):
        # share u+v
        num_share_para += 3*o[i]*rank_c[i] *2

    num_indep_para = 0 
    for i in range(len(o)):
        num_indep_para += 6*o[i] * int(o[i] * ri_rate) * (2*repeat_num[i] -1)

    num_svd_para = 0
    for i in range(len(o)):
        num_svd_para += 3*o[i]*int(o[i]*rank_rate_svd) + 3*(o[i]/2)*int(o[i]*rank_rate_svd)

    num = num_share_para + num_indep_para + num_svd_para

    if print_ph:
        print(num)

    return num 


print('Rank_rate_svd')
print(Rank_rate_svd_str)
print(Rank_rate_svd)
print(np.array(Rank_rate_svd)*128)

print('\n')
print('Rc_rate')
print(Rc_rate_str)
print(Rc_rate)
print(np.array(Rc_rate)*128)


Ri_rate = []
for rank_rate_svd in Rank_rate_svd:
    Ri_rate_each_r_svd = []
    for rc_rate in Rc_rate:

        ri_rate = count_csvd_ri(rank_rate_svd, rc_rate)
        # print('rank_rate_svd  ri_rate: ', rank_rate_svd, ri_rate)

        Ri_rate_each_r_svd.append(ri_rate)
    Ri_rate.append(Ri_rate_each_r_svd)
print('\nRi_rate')
for i in Ri_rate:
    print(i)
Ri_rate_2 = np.array(Ri_rate.copy())
Ri_rate_2[Ri_rate_2<0]=0
print(Ri_rate_2)



print('\nRi')
for i in Ri_rate:
    print([0 if element<0 else int(element*128) for element in i])


#ctd参数量

Num_ctd_collect = []
for i in range(len(Rank_rate_svd)):
    Num_csvd = []
    for j in range(len(Rc_rate)):
        num = num_csvd(Rank_rate_svd[i], Rc_rate[j],Ri_rate[i][j])
        Num_csvd.append(num)
    Num_ctd_collect.append(Num_csvd)
print('num of cvsd')
for i in range(len(Rank_rate_svd)):
    print(Num_ctd_collect[i])

print('\nnum_svd')
#svd参数量
Num_svd = []
for i in Rank_rate_svd:
    Num_svd.append(num_svd(i))
print(Num_svd)


print('\n csvd CR')
num_other_parameter = other_parameter()
CR_pcsvd = ((orig()+num_other_parameter) /(np.array(Num_ctd_collect) + num_other_parameter))
CR_pcsvd[np.array(Num_ctd_collect)==0]=0
print(CR_pcsvd)
print('\n CR svd')
print((orig()+num_other_parameter) / (np.array(Num_svd) + num_other_parameter))
# para_tt = np.array(para_tt)
# para_ctd = np.array(para_ctd)
# num_other_parameter = other_parameter()

# print('\nRatio')
# print((para_tt+num_other_parameter)/(orig()+num_other_parameter))
# print((para_ctd+num_other_parameter)/(orig()+num_other_parameter))


# print('\nCR')
# print(1/((para_tt+num_other_parameter)/(orig()+num_other_parameter)))
# print(1/((para_ctd+num_other_parameter)/(orig()+num_other_parameter)))

# # print(other_parameter())
# # print(orig())

# #ncptd 结果
# r_q = np.array([3/8,1/2, 5/8])*128
# print(r_q)

# print(orig()+num_other_parameter)
