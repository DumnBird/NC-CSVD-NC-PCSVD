import numpy as  np 


#resnet8
in_out = [[64,128],[128,128],[128,256],[256,256],[256,512],[512,512]]

Rank_rate_svd_str = ['1/32','1/16','1/8','1/4', '3/8', '1/2', '3/4']
Rank_rate_svd = [eval(i) for i in Rank_rate_svd_str]


Rc_rate_str = ['1/64','1/32','1/16','1/8', '1/4','1/2']
Rc_rate = [eval(i) for i in Rc_rate_str]




#paramter in shorcut downsample layer  and the 3*3*3*64layer
def other_parameter():
    num = 3*3*3*64 + 64*128 +128*256 +256*512+512*10+10
    return num

#num of  parameters of original resnet8
def orig():
    num = 0
    for i in range(len(in_out)):
        num += 3*3*in_out[i][0]*in_out[i][1] 
    return num 

#num of parameter in the svd model under rank_rate_svd
def num_svd(rank_rate_svd):

    Rank_svd = [int(in_out[i][1]*rank_rate_svd) for i in range(len(in_out))]

    num = 0
    for i in range(len(in_out)):
        #filter = 3*3
        num += 3 * in_out[i][0] * Rank_svd[i] + 3 * in_out[i][1] * Rank_svd[i]
    return num

#With rank_rate_svd, rc_rate, calculate ri_rate
def count_pcsvd_ri(rank_rate_svd, rc_rate, print_ph=0):

    #base on ouput depth
    rank_c  = [int(in_out[i][1] * rc_rate) for i in range(len(in_out))]

    num_share_para = 0
    for i in range(3):
        i = i * 2  #0,2,4
        #u_pshare1 +u_pshare2 +v_share
        num_share_para += (3*in_out[i][0] + 3*in_out[i][1]) * rank_c[i] + rank_c[i] * 3*in_out[i][1]

    num_indep_para = num_svd(rank_rate_svd) - num_share_para

    #num_indep_para = sum[(3*I + 3*O) * (O*ri_rate)]=sum[(I + O) * 3O]*ri_rate--->
    coefficient = 0 
    for i in range(len(in_out)):
        coefficient += (in_out[i][0] + in_out[i][1]) * 3 * in_out[i][1]
    ri_rate = num_indep_para / coefficient

    if print_ph:
        print(ri_rate)

    return ri_rate


def num_pcsvd(rc_rate, ri_rate, print_ph=False):
    if ri_rate<0:
        return 0
    #base on ouput depth

    rank_c  = [int(in_out[i][1] * rc_rate) for i in range(len(in_out))]

    num_share_para = 0
    for i in range(3):
        i = i * 2  #0,2,4
        #u_pshare1 +u_pshare2 +v_share
        num_share_para += (3*in_out[i][0] + 3*in_out[i][1]) * rank_c[i] + rank_c[i] * 3*in_out[i][1]

    num_indep_para = 0 
    for i in range(len(in_out)):
        num_indep_para += (3*in_out[i][0] + 3*in_out[i][1]) * int(in_out[i][1] * ri_rate)

    num = num_share_para + num_indep_para

    if print_ph:
        print(num)

    return num 





Ri_rate = []
for rank_rate_svd in Rank_rate_svd:
    Ri_rate_each_r_svd = []
    for rc_rate in Rc_rate:

        ri_rate = count_pcsvd_ri(rank_rate_svd, rc_rate)
        print('rank_rate_svd  ri_rate: ', rank_rate_svd, ri_rate)

        Ri_rate_each_r_svd.append(ri_rate)
    Ri_rate.append(Ri_rate_each_r_svd)

print('Rank_rate_svd')
print(Rank_rate_svd_str)
print(Rank_rate_svd)
print(np.array(Rank_rate_svd)*128)

print('\n')

print('Rc_rate')
print(Rc_rate_str)
print(Rc_rate)
print(np.array(Rc_rate)*128)



print('\nRi_rate')
for i in Ri_rate:
    print(i)
Ri_rate_2 = np.array(Ri_rate.copy())
Ri_rate_2[Ri_rate_2<0]=0
print(Ri_rate_2)



print('\nRi')
for i in Ri_rate:
    print([0 if element<0 else int(element*128) for element in i])


#amount of parameters in pctd 

Num_pctd_collect = []
for i in range(len(Rank_rate_svd)):
    Num_pcsvd = []
    for j in range(len(Rc_rate)):
        num = num_pcsvd(Rc_rate[j],Ri_rate[i][j])
        Num_pcsvd.append(num)
    Num_pctd_collect.append(Num_pcsvd)
print('num of pcvsd')
for i in range(len(Rank_rate_svd)):
    print(Num_pctd_collect[i])

print('\nnum_svd')
#svd amount
Num_svd = []
for i in Rank_rate_svd:
    Num_svd.append(num_svd(i))
print(Num_svd)


print('\n pcsvd CR')
num_other_parameter = other_parameter()
CR_pcsvd = ((orig()+num_other_parameter) /(np.array(Num_pctd_collect) + num_other_parameter))
CR_pcsvd[np.array(Num_pctd_collect)==0]=0
print(CR_pcsvd)
# print(((orig()+num_other_parameter) /(np.array(Num_pctd_collect) + num_other_parameter))[Num_pctd_collect==0]=0)
print('\n CR svd')
print((orig()+num_other_parameter) / (np.array(Num_svd) + num_other_parameter))
# para_tt = np.array(para_tt)
# para_pctd = np.array(para_pctd)
# num_other_parameter = other_parameter()

# print('\nRatio')
# print((para_tt+num_other_parameter)/(orig()+num_other_parameter))
# print((para_pctd+num_other_parameter)/(orig()+num_other_parameter))


# print('\nCR')
# print(1/((para_tt+num_other_parameter)/(orig()+num_other_parameter)))
# print(1/((para_pctd+num_other_parameter)/(orig()+num_other_parameter)))

# # print(other_parameter())
# # print(orig())

# #ncptd outcome
# r_q = np.array([3/8,1/2, 5/8])*128
# print(r_q)

# print(orig()+num_other_parameter)
