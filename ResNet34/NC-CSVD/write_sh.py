import numpy as np


Rank_rate_svd = [0.1875, 0.25, 0.375, 0.5, 0.75]
R_c= [0.0625, 0.125, 0.1875, 0.25, 0.5]


R_i = [[0.17748091603053434, 0.1674618320610687, 0.15744274809160305, 0, 0],
       [0.23998091603053434, 0.2299618320610687, 0.21994274809160305, 0, 0],
       [0.36498091603053434, 0.3549618320610687, 0.3449427480916031, 0, 0],
       [0.48998091603053434, 0.4799618320610687, 0.4699427480916031, 0.4599236641221374, 0],
       [0.7399809160305344, 0.7299618320610687, 0.7199427480916031, 0.7099236641221374, 0.6698473282442748]]
print('Rc')
print([int(item*128) for item in R_c])

print('\nRi')
for item in R_i:
  print([int(i*128) for i in item])



count1 = 0
count2 = 0
count3 = 0
for ci in range(len(R_i)):
    for cc in range(len(R_c)):
        lr = 4
        r_svd = Rank_rate_svd[ci]
        rc_rate = R_c[cc]
        ri_rate = R_i[ci][cc]

        count1 += 1       

        if ri_rate<=0:
            pass
        else:
            count2 += 1
            warm_up_option='False'
            if ci==0 or ci==1:
              lr=3

            print('python ../resnet34_csvd.py --rc=%s --ri=%s --rank_rate_svd=%s --iter=9 --warm_up=%s --ckpt_path=20201010_group%d_small_lr --num_lr=1e-%d --gpu=%d '%(str(rc_rate), str(ri_rate), str(r_svd), warm_up_option, ci+1, lr, (count2//3)+2)) 
            if warm_up_option=='False':
                count3+=1 
        
print(count1)
print(count2)
print(count3)
