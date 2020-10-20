import numpy as np
import os
import tensorflow as tf
import time
from tensorflow.python import pywrap_tensorflow


flags = tf.flags
FLAGS=flags.FLAGS
flags.DEFINE_string('train_set_number_rate', '1', 'how many train_set used to train')
flags.DEFINE_string('epoch', '4', 'train epoch' )

flags.DEFINE_string('rank_rate', '3/4', "rank rate tt")
flags.DEFINE_string('gpu', '2', 'gpu choosed to used' )
flags.DEFINE_string('num_lr', '1e-3', 'initial learning_rate')

flags.DEFINE_string('warm_up', 'False', "rank_rate")
flags.DEFINE_string('warm_up_init_lr', '1e-5','initial learning rate when warm up')
flags.DEFINE_string('warm_up_epoch','30','how many fine-tune epoch when warm up is used')

flags.DEFINE_string('ckpt_path', '20201009_all_train_set', 'ckpt path to restore' )
ckpt_path = './'+FLAGS.ckpt_path



os.environ['CUDA_VISIBLE_DEVICES']=FLAGS.gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True




import sys
sys.path.append("/home/test01/csw/resnet34")
import imagenet_data
import image_processing
import math





imagenet_data_val = imagenet_data.ImagenetData('validation')
imagenet_data_train = imagenet_data.ImagenetData('train', eval(FLAGS.train_set_number_rate))

val_images, val_labels = image_processing.inputs(imagenet_data_val, batch_size=500, num_preprocess_threads=16)
#256 is too large for linux to train---OOM
train_images, train_labels =  image_processing.inputs(imagenet_data_train, batch_size=128, num_preprocess_threads=16)




#G1 means U_4d, G2 means V_4d
def decompose(weight, rank):
    F1, F2, I, O = weight.shape  # weight[F1,F2,I,O]

    assert(rank==int(O*eval(FLAGS.rank_rate)))
    W = np.reshape(np.transpose(weight,(0,2,1,3)), [F1*I, -1])
    u1, s1, v1 = np.linalg.svd(W, full_matrices=True)
    G1  = np.dot(u1[:, 0:rank].copy(),np.diag(s1)[0:rank, 0:rank].copy())
    G1 = np.transpose(np.reshape(G1, [F1, I, 1, rank]),(0,2,1,3))
    G2 = v1[0:rank, :].copy()
    G2 = np.transpose(np.reshape(G2,[1,rank,F2,O]),(0,2,1,3))

    return G1,G2

def basic_conv(in_tensor, value_dict, layer_num, repeat_num, conv_num, flag_shortcut, down_sample, is_training, res_in_tensor=None, rank_rate=None):


    if layer_num != 1 and repeat_num == 0 and conv_num == 1:
        strides = [1, 2, 2, 1]
    else:
        strides = [1, 1, 1, 1]

    # keep 64*64 layerthe same；
    if layer_num==1 :
        name_weight = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.'  + 'conv' + str(conv_num) + '.weight'
        weight = tf.Variable(value_dict[name_weight], name=name_weight)
        tensor_go_next = tf.nn.conv2d(in_tensor, weight, strides, padding='SAME')

    #SVD
    else:
        name_scope = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.'
        name_weight = name_scope + 'conv' + str(conv_num) + '.weight'

        name_weight1 = name_weight + 'U'
        name_weight2 = name_weight + 'V'

        shape = value_dict[name_weight].shape

        rank = int(shape[3]*rank_rate)

        os.makedirs(ckpt_path+'/rank_rate'+str(FLAGS.rank_rate[0])+'_'+str(FLAGS.rank_rate[2:])+'/', exist_ok=True)
        with open(log_path, 'a') as f:
            f.write(name_scope + ' ratio=' + FLAGS.rank_rate + ' rank=' + str(rank) + '\n')

        U,V = decompose(value_dict[name_weight],rank)
        # SVD convert the original Conv layer into 2 independent Conv layers
        weight1 = tf.Variable(U, name=name_weight1)
        tensor_go_next = tf.nn.conv2d(in_tensor, weight1, [1, strides[1], 1, 1], padding='SAME')

        weight2 = tf.Variable(V, name=name_weight2)
        tensor_go_next = tf.nn.conv2d(tensor_go_next, weight2, [1, 1, strides[2], 1], padding='SAME')


        #fine-tune decomposed layers
        var_list_to_train.append(weight1)
        var_list_to_train.append(weight2)



    name_scope = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.'
    name_bn_scope = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.' + 'bn' + str(conv_num)

    # THE SAME
    tensor_go_next = tf.contrib.layers.batch_norm(tensor_go_next, decay=0.9, center=True, scale=True, epsilon=1e-9,
                                                  updates_collections=tf.GraphKeys.UPDATE_OPS,
                                                  is_training=is_training, scope=name_bn_scope)

    print('shape of tensor_go_next', tensor_go_next.shape)
    print('\n\n')

    if flag_shortcut == True:
        if down_sample == True:

            name_downsample_weight = name_scope + 'downsample.0.weight'
            name_downsample_bn_scope = name_scope + 'downsample.1'

            weight_down_sample = tf.Variable(value_dict[name_downsample_weight], name=name_downsample_weight)
            shorcut_tensor = tf.nn.conv2d(res_in_tensor, weight_down_sample, [1, 2, 2, 1], padding='SAME')
            shorcut_tensor = tf.contrib.layers.batch_norm(shorcut_tensor, decay=0.9, center=True, scale=True,
                                                          epsilon=1e-9, updates_collections=tf.GraphKeys.UPDATE_OPS,
                                                          is_training=is_training, scope=name_downsample_bn_scope)
        else:
            shorcut_tensor = res_in_tensor

        tensor_go_next = tensor_go_next + shorcut_tensor

    tensor_go_next = tf.nn.relu(tensor_go_next)

    return tensor_go_next

def basic_block(in_tensor, value_dict, layer_num, repeat_num, down_sample, is_training=False, rank_rate=None):



    tensor_go_next =  basic_conv(in_tensor, value_dict, layer_num, repeat_num, conv_num=1,
                                     is_training=is_training,flag_shortcut=False, down_sample=False,  res_in_tensor= None, rank_rate=rank_rate)
    tensor_go_next =  basic_conv(tensor_go_next, value_dict, layer_num, repeat_num, conv_num=2,
                                     is_training=is_training, flag_shortcut=True, down_sample=down_sample,  res_in_tensor = in_tensor, rank_rate=rank_rate)
    return tensor_go_next

def repeat_basic_block(in_tensor, value_dict, repeat_times, layer_num, is_training, rank_rate=None):

    tensor_go_next = in_tensor
    for i in range(repeat_times):
        down_sample = True if i == 0 and layer_num != 1 else False
        # if i==0:
        #     tensor_go_next = orig_basic_block(tensor_go_next, value_dict, layer_num, repeat_num=i, down_sample=down_sample, is_training=is_training)
        # else:

        tensor_go_next = basic_block(tensor_go_next, value_dict, layer_num, repeat_num=i, down_sample=down_sample, is_training=is_training, rank_rate=rank_rate)
    return tensor_go_next

def resnet34(x, value_dict, is_training, rank_rate):


    weight_layer1 = tf.Variable(value_dict['conv1.weight'], dtype = tf.float32, name='conv1.weight')
    name_bn1_scope = 'bn1'
    tensor_go_next = tf.nn.conv2d(x, weight_layer1,[1,2,2,1], padding='SAME')

    tensor_go_next = tf.contrib.layers.batch_norm(tensor_go_next, decay = 0.9, center = True, scale = True,epsilon=1e-9, updates_collections=tf.GraphKeys.UPDATE_OPS,
                                                        is_training=is_training, scope=name_bn1_scope)
    tensor_go_next = tf.nn.relu(tensor_go_next)

    tensor_go_next = tf.nn.max_pool(tensor_go_next, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')


    '''64 block1不分解'''
    tensor_go_next = repeat_basic_block(tensor_go_next, value_dict, 3, 1, is_training, rank_rate=None)
    print('\nblock1 finished')


    tensor_go_next = repeat_basic_block(tensor_go_next, value_dict, 4, 2, is_training, rank_rate)
    print('\nblock2 finished')

    tensor_go_next = repeat_basic_block(tensor_go_next, value_dict, 6, 3, is_training, rank_rate)
    print('\nblock3 finished')

    tensor_go_next = repeat_basic_block(tensor_go_next, value_dict, 3, 4, is_training, rank_rate)
    print('\nblock4 finished')


    tensor_go_next = tf.layers.average_pooling2d(tensor_go_next, pool_size=7, strides=1)

    tensor_go_next_flatten = tf.reshape(tensor_go_next, [-1,512], name='tensor_go_next_flatten')
    weight_fc = tf.Variable(value_dict['fc.weight'], dtype=tf.float32, name='fc.weight')
    bias_fc = tf.Variable(value_dict['fc.bias'], dtype=tf.float32, name='fc.bias')
    tensor_go_next_fc = tf.matmul(tensor_go_next_flatten, weight_fc) + bias_fc

    output = tf.nn.softmax(tensor_go_next_fc, name='softmax')

    return tensor_go_next_fc, output


var_list_to_train = []  
rank_rate = eval(FLAGS.rank_rate)

#get orig value_dict
reader = pywrap_tensorflow.NewCheckpointReader('/home/test01/csw/resnet34/ckpt3_20200617/resnet-34-accTensor("accuracy:0", shape=(), dtype=float32)-5')  # tf.train.NewCheckpointReader
var_to_shape_map = reader.get_variable_to_shape_map()
value_dict = {}
for key in var_to_shape_map:
    value = reader.get_tensor(key)
    value_dict[key] = value


ckpt_root_path = ckpt_path+'/rank_rate'+str(FLAGS.rank_rate[0])+'_'+str(FLAGS.rank_rate[2:])
os.makedirs(ckpt_root_path, exist_ok=True)
log_path = ckpt_root_path+ '/'+str(FLAGS.rank_rate[0])+'_'+str(FLAGS.rank_rate[2])+'log.log'





lr = tf.placeholder(tf.float32, name='learning_rate')
is_training = tf.placeholder(tf.bool, name = 'is_training')
x = tf.placeholder(tf.float32, [None,224,224,3], name = 'x')
y = tf.placeholder(tf.int64, [None], name="y")


print('start building network')
tensor_go_next_fc, output = resnet34(x, value_dict, is_training, rank_rate)
print('building network done\n\n')



loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tensor_go_next_fc, labels=y), name='loss')

optimizer = tf.train.MomentumOptimizer(learning_rate = lr, momentum=0.9, name='Momentum' )
train_op = optimizer.minimize(loss, name = 'train_op', var_list = var_list_to_train)




correct_predict = tf.equal(tf.argmax(output, 1), y, name = 'correct_predict')
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32), name="accuracy")



with tf.Session(config = config) as sess:
    sess.run(tf.global_variables_initializer())
    graph = tf.get_default_graph()


    '''
    #assign original parameters of BN layers
    #But I found that these parameters actually can be assigned directly with other BN API：
    tf.layers.batch_normalization(tensor_go_next,  
                                                beta_initializer=tf.constant_initializer(value_dict[name_bn_scope+"/beta"]),
                                                gamma_initializer=tf.constant_initializer(value_dict[name_bn_scope+"/gamma"]),
                                                moving_mean_initializer=tf.constant_initializer(value_dict[name_bn_scope+"/moving_mean"]),
                                                moving_variance_initializer=tf.constant_initializer(value_dict[name_bn_scope+"/moving_variance"]),
                                                training=is_training, name=name_bn_scope)
    '''

    print('start assign \n')
    sess.run(tf.assign(graph.get_tensor_by_name('bn1/moving_mean:0'), value_dict['bn1/moving_mean']))
    sess.run(tf.assign(graph.get_tensor_by_name('bn1/moving_variance:0'), value_dict['bn1/moving_variance']))
    print('\n\nbn1 var assign finished ')
    sess.run(tf.assign(graph.get_tensor_by_name('bn1/gamma:0'), value_dict['bn1/gamma']))
    print('\n\nbn1 gamma assign finished ')

    sess.run(tf.assign(graph.get_tensor_by_name('bn1/beta:0'), value_dict['bn1/beta']))
    print('\n\nbn1 beta  assign finished ')


    for num in [[1,3],[2,4],[3,6],[4,3]]:
        for num_repeat in range(num[1]):
            name_scope = 'layer'+str(num[0])+'.'+str(num_repeat)+'.bn1'
            tf_name_scope = name_scope + '/'
            for name in [['gamma','gamma:0'], ['beta','beta:0'], ['moving_mean', 'moving_mean:0'], ['moving_variance', 'moving_variance:0']]:
                sess.run(tf.assign(graph.get_tensor_by_name(tf_name_scope+name[1]), value_dict[tf_name_scope+name[0]]))
                print(tf_name_scope + name[1])
            name_scope = 'layer'+str(num[0])+'.'+str(num_repeat)+'.bn2'
            tf_name_scope = name_scope + '/'
            for name in [['gamma','gamma:0'], ['beta','beta:0'], ['moving_mean', 'moving_mean:0'], ['moving_variance', 'moving_variance:0']]:
                print(tf_name_scope+name[1])
                sess.run(tf.assign(graph.get_tensor_by_name(tf_name_scope+name[1]), value_dict[tf_name_scope+name[0]]))

    for num in [2,3,4]:
        name_downsample_scope = 'layer'+str(num)+'.0.downsample.1'

        for name in [['gamma','gamma:0'], ['beta','beta:0'], ['moving_mean', 'moving_mean:0'], ['moving_variance', 'moving_variance:0']]:
            sess.run(tf.assign(graph.get_tensor_by_name(name_downsample_scope +'/'+ name[1]), value_dict[name_downsample_scope +'/'+ name[0]]))
    print('\n\nAssign done\n\n')


    coord = tf.train.Coordinator()
    threads = []
    for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
    print('coord done \n')

    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    bn_moving_vars += [g for g in g_list if 'Momentum' in g.name]
    var_list += bn_moving_vars

    saver = tf.train.Saver(var_list=var_list, max_to_keep=10)
    saver.save(sess, ckpt_root_path+'/single_tt_raw/'+'single_tt_raw', write_meta_graph=True)

    try:

        test_acc = 0
        test_count = 0
        start = time.time()
        print('\n\nstart validation\n\n')
        for i in range(int(50000 / 500)):

            image_batch, label_batch = sess.run([val_images, val_labels])
            acc= sess.run(accuracy, feed_dict={x: image_batch, y: label_batch - 1, is_training:False})
            test_acc += acc
            test_count += 1
        test_acc /= test_count
        print("\n\nTest Accuracy: \n " + str(test_acc))


        with open(log_path, 'a') as f:
            f.write('rank_rate: ' + FLAGS.rank_rate)
            f.write('\naccuracy_WithOut_Train: ' + str(test_acc)+'\n\n')
            f.write('\nhow much train set to be used: '+ FLAGS.train_set_number_rate+'\n')
            f.write('wam up or not: '+FLAGS.warm_up+'\n')
            f.write('epoch: ' + (FLAGS.epoch if FLAGS.warm_up=='False' else FLAGS.warm_up_epoch)  + '\n')


        print('how much train set to be used: ', FLAGS.train_set_number_rate)

        ave_time = 0
        epoch = eval(FLAGS.epoch if FLAGS.warm_up=='False' else FLAGS.warm_up_epoch)
        print('\n\nstart training\n')
        for j in range(epoch):
            print('j=', j)
            print('\n\nstart training')

            if eval(FLAGS.warm_up):
                if j==0:
                    num_lr = eval(FLAGS.warm_up_init_lr)
                    print('learning_rate:', num_lr)

                if j==1:
                    num_lr = 0.001
                if j==20:
                    num_lr = 0.001/10
 
            else:    
                if j == 0:
                    num_lr = eval(FLAGS.num_lr)
                    print('initial learning_rate: ', num_lr, '\n')
                else:
                    num_lr = num_lr/10

            # if j==0:
            #     print('initial learning_rate: ', num_lr,'\n')
            # elif j%2==0:
            #     num_lr = num_lr / 10 
            #     print('learning_rate: ', num_lr, '\n')


            if FLAGS.train_set_number_rate=='1':
                n = 1281167
            else:
                n=round(1281167*eval(FLAGS.train_set_number_rate))  #训练图片数量
            for i in range(int(n/128)):

                image_batch, label_batch = sess.run([train_images, train_labels])

                '''is_trianing 必须设为False, 不然bn的mean 和variance会变'''
                _, loss_eval = sess.run([train_op, loss], feed_dict={x: image_batch, y: label_batch - 1, is_training: False, lr:num_lr})
                if i==0 and j==0:
                    first_loss = loss_eval
                if i%200 ==0:
                    print('j='+str(j)+' i='+str(i)+ '  loss='+str(loss_eval))




            test_acc = 0
            test_count = 0
            time_count = 0

            print('\nstart validation ,j=', j)
            for i in range(int(50000 / 500)):

                image_batch, label_batch = sess.run([val_images, val_labels])

                start = time.time()
                acc = sess.run(accuracy, feed_dict={x: image_batch, y: label_batch - 1, is_training: False})
                end = time.time()

                time_count += end-start
                test_acc += acc
                test_count += 1
            test_acc /= test_count
            time_ave_one_epoch = time_count/ 50  #/50000*1000=/50 ms per image
            print(time_ave_one_epoch)

            ave_time += time_ave_one_epoch / epoch

            print("Test Accuracy:  " + str(test_acc))


            with open(log_path, 'a') as f:
                if j==0:
                    f.write('rank_rate:'+str(FLAGS.rank_rate)+'\n')
                    f.write('Num_Train_picture: '+str(n)+'\n')
                    f.write('first loss: ' + str(first_loss)+'\n')

                f.write('\nepoch= '+ str(j)+'\n')
                f.write('learning_rate: '+str(num_lr)+'\n')
                f.write('final loss: ' + str(loss_eval)+'\n')
                f.write('accuracy_after_this_Train: ' + str(test_acc)+'\n\n')
                f.write('time_ave_one_epoch : ' + str(time_ave_one_epoch) + '\n\n')
                   

                if j+1==epoch:
                    f.write('time_ave_all_epoch : ' + str(ave_time) + 'ms gpu per imag\n\n')
                    print(ave_time)
                    saver.save(sess, ckpt_root_path+'/resnet34svd-acc'+str(ave_time)[0:7]+'-ave3time'+str(ave_time)[0:7], global_step=j+1)
                else:
                    saver.save(sess, ckpt_root_path+'/resnet34svd-acc'+str(test_acc)[0:7]+'-time'+str(time_ave_one_epoch)[0:7], global_step=j+1)


    finally:
            coord.request_stop()
            coord.join(threads)



