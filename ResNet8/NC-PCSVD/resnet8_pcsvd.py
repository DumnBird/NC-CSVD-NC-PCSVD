import tensorflow as tf
import numpy as np
import os
import cifar10_input

import time




flags = tf.flags
FLAGS=flags.FLAGS
flags.DEFINE_string('epoch', '15', 'train epoch' )


flags.DEFINE_string('rc', '3/4', "rank ratio of shared part")
flags.DEFINE_string('ri', '3/4', "rank ratio of independent part")
flags.DEFINE_string('iter', '0', 'iter number of npy')

flags.DEFINE_string('gpu', '0', 'gpu choosed to used' )
flags.DEFINE_string('num_lr', '1e-3', 'initial learning_rate')
flags.DEFINE_string('better_care','False','fine-tune setting for very high performance')
flags.DEFINE_string('warm_up', 'False', "rank_rate")
flags.DEFINE_string('warm_up_init_lr', '1e-5','initial learning rate when warm up')
flags.DEFINE_string('warm_up_epoch','30','how many fine-tune epoch when warm up is used')

flags.DEFINE_string('ckpt_path', '20201009_inilr_1e-4', 'ckpt path to restore' )
ckpt_path = './'+FLAGS.ckpt_path


os.environ['CUDA_VISIBLE_DEVICES']=FLAGS.gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


#partly share component
def basic_conv_partly_shared(in_tensor, layer_num, repeat_num, conv_num,stride):
    #U_partly_share
    name_weight1 = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.conv' + str(conv_num) + '.U_partly_shared'

    name_weight2 = 'layer' + str(layer_num) + '.conv' + str(conv_num) + '.V_shared'

    #SVD convert the original Conv layer into 2 independent Conv layers
    weight1 = tf.get_variable(name=name_weight1, dtype=tf.float32, initializer=tf.constant(proposed_dict[name_weight1]), regularizer=regularizer)
    tensor_go_next = tf.nn.conv2d(in_tensor, weight1, [1, stride, 1, 1], padding='SAME')

    weight2 = tf.get_variable(name=name_weight2, dtype=tf.float32, initializer=tf.constant(proposed_dict[name_weight2]), regularizer=regularizer)
    tensor_go_next = tf.nn.conv2d(tensor_go_next, weight2, [1, 1, stride, 1], padding='SAME')

    #fine-tune decomposed layers
    var_list_to_train.append(weight1)
    var_list_to_train.append(weight2)
    return tensor_go_next


#independent commponet
def basic_conv_independent(in_tensor, layer_num, repeat_num, conv_num,stride):
    name_scope = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.'
    name_weight = name_scope + 'conv' + str(conv_num) + '.'

    name_weight1 = name_weight + 'U'
    name_weight2 = name_weight + 'V'

    #SVD convert the original Conv layer into 2 independent Conv layers
    weight1 = tf.get_variable(name=name_weight1, dtype=tf.float32, initializer=tf.constant(proposed_dict[name_weight1]), regularizer=regularizer)
    tensor_go_next = tf.nn.conv2d(in_tensor, weight1, [1, stride, 1, 1], padding='SAME')
    
    weight2 = tf.get_variable(name=name_weight2, dtype=tf.float32, initializer=tf.constant(proposed_dict[name_weight2]), regularizer=regularizer)
    tensor_go_next = tf.nn.conv2d(tensor_go_next, weight2, [1, 1, stride, 1], padding='SAME')

    #fine-tune decomposed layers
    var_list_to_train.append(weight1)
    var_list_to_train.append(weight2)

    return tensor_go_next


 

def basic_conv(in_tensor, in_channels, out_channels, stride,  layer_num, repeat_num, conv_num, is_training, res_in_tensor=None):


    name_scope = 'layer'+ str(layer_num) + '.'+str(repeat_num) + '.'
    name_weight = name_scope + 'conv' + str(conv_num) + '.weight'
    name_bn_scope = name_scope + 'bn' + str(conv_num)

    independ_component = basic_conv_independent(in_tensor, layer_num, repeat_num, conv_num,stride)
    shared_component = basic_conv_partly_shared(in_tensor, layer_num, repeat_num, conv_num,stride)

    tensor_go_next = independ_component + shared_component

    tensor_go_next = tf.layers.batch_normalization(tensor_go_next,  
                                                beta_initializer=tf.constant_initializer(value_dict[name_bn_scope+"/beta"]),
                                                gamma_initializer=tf.constant_initializer(value_dict[name_bn_scope+"/gamma"]),
                                                moving_mean_initializer=tf.constant_initializer(value_dict[name_bn_scope+"/moving_mean"]),
                                                moving_variance_initializer=tf.constant_initializer(value_dict[name_bn_scope+"/moving_variance"]),
                                                training=is_training, name=name_bn_scope)


    # if conv_num==2:
    assert(conv_num==1)
    if conv_num==1:    
        assert(res_in_tensor != None)
        if layer_num!=1 and repeat_num==0:
            name_downsample_weight = name_scope + 'downsample.0.weight'
            name_downsample_bn_scope = name_scope + 'downsample.1'
                                
            weight_downsample = tf.get_variable(name=name_downsample_weight, dtype=tf.float32, initializer=tf.constant(value_dict[name_downsample_weight]), regularizer=regularizer)
            shortcut = tf.nn.conv2d(res_in_tensor, weight_downsample, strides=[1, 2, 2, 1], padding='SAME')

            shortcut = tf.layers.batch_normalization(shortcut,  
                                                beta_initializer=tf.constant_initializer(value_dict[name_downsample_bn_scope+"/beta"]),
                                                gamma_initializer=tf.constant_initializer(value_dict[name_downsample_bn_scope+"/gamma"]),
                                                moving_mean_initializer=tf.constant_initializer(value_dict[name_downsample_bn_scope+"/moving_mean"]),
                                                moving_variance_initializer=tf.constant_initializer(value_dict[name_downsample_bn_scope+"/moving_variance"]),
                                                training=is_training, name=name_downsample_bn_scope)


        else:
            shortcut = res_in_tensor

        tensor_go_next = tensor_go_next + shortcut

    tensor_go_next = tf.nn.relu(tensor_go_next)
    print('\n' + name_scope + 'conv' + str(conv_num))
    print(tensor_go_next.shape)
    return tensor_go_next

def basic_block(in_tensor, in_channels, out_channels, layer_num, repeat_num, is_training):
    stride = 1
    if layer_num!=1 and repeat_num==0:
        stride=2
    tensor_go_next = basic_conv(in_tensor, in_channels, out_channels, stride, layer_num, repeat_num, 1, is_training, res_in_tensor=in_tensor)
    # tensor_go_next = basic_conv(in_tensor, in_channels, out_channels, stride, layer_num, repeat_num, 1, is_training)
    # tensor_go_next = basic_conv(tensor_go_next, in_channels, out_channels, 1, layer_num, repeat_num, 2, is_training, res_in_tensor=in_tensor)
    return tensor_go_next

def repeat_basic_block(in_tensor, in_channels, out_channels, layer_num, repeat_times,is_training):
    tensor_go_next = in_tensor
    with tf.variable_scope('', reuse=tf.AUTO_REUSE):
        for i in range(repeat_times):
            print('\n\n'+str(i))
            tensor_go_next = basic_block(tensor_go_next, in_channels, out_channels, layer_num, i, is_training)
    return tensor_go_next

def resnet8(in_tensor, is_training):
    weight_layer1 = tf.get_variable(name='conv1.weight', dtype=tf.float32, initializer=tf.constant(value_dict['conv1.weight']), regularizer=regularizer)
    tensor_go_next = tf.nn.conv2d(in_tensor, weight_layer1,strides=[1,1,1,1], padding='SAME')


    tensor_go_next = tf.layers.batch_normalization(tensor_go_next,  
                                                beta_initializer=tf.constant_initializer(value_dict['bn1'+"/beta"]),
                                                gamma_initializer=tf.constant_initializer(value_dict['bn1'+"/gamma"]),
                                                moving_mean_initializer=tf.constant_initializer(value_dict['bn1'+"/moving_mean"]),
                                                moving_variance_initializer=tf.constant_initializer(value_dict['bn1'+"/moving_variance"]),
                                                training=is_training, name='bn1')

    tensor_go_next = tf.nn.relu(tensor_go_next)

    #no maxpool
    # tensor_go_next = tf.nn.max_pool(tensor_go_next, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

    #No 3*364*64 layers
    # tensor_go_next = repeat_basic_block(tensor_go_next, 64,  64, layer_num=1, repeat_times=2, is_training=is_training)
    # print('\nblock1 finished')

    tensor_go_next = repeat_basic_block(tensor_go_next, 64, 128, layer_num=2, repeat_times=2, is_training=is_training)
    print('\nblock2 finished')

    tensor_go_next = repeat_basic_block(tensor_go_next, 128, 256, layer_num=3, repeat_times=2,  is_training=is_training)
    print('\nblock3 finished')

    tensor_go_next = repeat_basic_block(tensor_go_next, 256, 512, layer_num=4, repeat_times=2, is_training=is_training)  #(4,4,512)
    print('\nblock4 finished')

    tensor_go_next = tf.layers.average_pooling2d(tensor_go_next, pool_size=4, strides=1) #(b,1,1,512)

    tensor_go_next_flatten = tf.reshape(tensor_go_next, [-1,512], name='tensor_go_next_flatten') #[b,512]


    weight_fc =  tf.get_variable(name='fc.weight', dtype=tf.float32, initializer=tf.constant(value_dict['fc.weight']), regularizer=regularizer)
    bias_fc =   tf.get_variable(name='fc.bias', dtype=tf.float32, initializer=tf.constant(value_dict['fc.bias']), regularizer=regularizer)


    tensor_go_next_fc = tf.matmul(tensor_go_next_flatten,weight_fc) + bias_fc

    output = tf.nn.softmax(tensor_go_next_fc, name='softmax')

    return tensor_go_next_fc, output




with tf.device('/cpu:0'):
    train_images, train_labels = cifar10_input.distorted_inputs(data_dir='/home/test01/csw/resnet18/cifar-10-batches-bin', batch_size= 256)
    test_images, test_labels = cifar10_input.inputs(data_dir='/home/test01/csw/resnet18/cifar-10-batches-bin', eval_data=True, batch_size=250)





'''Network'''
classes = ('plane', 'car', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck')
is_training = tf.placeholder(tf.bool,name='is_training')
x =  tf.placeholder(tf.float32, [None,32,32,3],name='x')
y =  tf.placeholder(tf.int64, [None],name='y')
lr = tf.placeholder(tf.float32, name='learning_rate')

initializer = tf.truncated_normal_initializer(stddev=0.01)
regularizer = tf.contrib.layers.l2_regularizer(5e-4) #L2 Regularizer

var_list_to_train = []

value_dict = np.load('/home/test01/csw/resnet8/tt/parameter.npy', allow_pickle=True).item()

rc = eval(FLAGS.rc)
ri = eval(FLAGS.ri)
proposed_dict = np.load('/home/test01/sambashare/sdd/ICASSP/resnet8/Decom/Widen/No_Relu/Iter/pcsvd_npy/iter%d_npy_rc=%f_ri=%f.npy'%(eval(FLAGS.iter),round(rc,4),round(ri,6)), allow_pickle=True).item()


ckpt_root_path = ckpt_path+'iter%d'%eval(FLAGS.iter)+ '/rc=%f_ri=%f'%(round(rc,4),round(ri,6))
os.makedirs(ckpt_root_path, exist_ok=True)
log_path = ckpt_root_path+ '/rc=%f_ri=%flog.log'%(round(rc,4),round(ri,6))


print('start building network')
logits, output = resnet8(x, is_training)
print('building network done\n\n')

keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) #L2 Regularizer
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.maximum(logits,1e-8), labels=y), name='loss') + tf.add_n(keys)
optimizer = tf.train.MomentumOptimizer(learning_rate=lr,momentum=0.9 , name='Momentum')
train_op = optimizer.minimize(loss, name = 'train_op', var_list = var_list_to_train)



correct_predict = tf.equal(tf.argmax(output, 1), y)
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32), name="accuracy")



with tf.Session(config=config) as sess:

    sess.run(tf.global_variables_initializer())

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





    saver = tf.train.Saver(var_list=var_list, max_to_keep=25)
    saver.save(sess, ckpt_root_path + '/single_tt_raw/raw', write_meta_graph=True)

    try:
        test_acc = 0
        test_count = 0
        start = time.time()
        for i in range(int(10000 / 250)) :

            image_batch, label_batch = sess.run([test_images, test_labels])

            acc = sess.run(accuracy, feed_dict={x: image_batch, y: label_batch, is_training: False})
            test_acc += acc
            test_count += 1
        test_acc /= test_count
        end = time.time()
        print("Test Accuracy before finetune:  " + str(test_acc))

        with open(log_path, 'a') as f:
            f.write('rc ri: ' + str(rc)+' ' +str(ri))
            f.write('\naccuracy_WithOut_Train: ' + str(test_acc)+'\n')
            f.write('time: ' + str(end-start)+'\n\n')


        '''Training'''
        f = open(log_path, 'a')

        f.write('time:2020/10/6  Reset8 for Cifar10 \n\n')
        f.write('iter: '+FLAGS.iter+'\n')
        f.write('initial learning_rate: ' + (FLAGS.warm_up_init_lr if eval(FLAGS.warm_up) else FLAGS.num_lr) + '\n')
        f.write('wam up or not: '+FLAGS.warm_up+'\n')
        f.write('epoch: ' + (FLAGS.epoch if FLAGS.warm_up=='False' else FLAGS.warm_up_epoch)  + '\n')

        ave_time = 0
        epoch = eval(FLAGS.epoch if FLAGS.warm_up=='False' else FLAGS.warm_up_epoch)
        print('\n\nstart training\n')
        for j in range(epoch):
            print('\n\nj= ', j)

            if eval(FLAGS.warm_up):
                if j==0:
                    num_lr = eval(FLAGS.warm_up_init_lr)
                    print('learning_rate:', num_lr)

                if j==1:
                    num_lr = 0.001
                if j==20:
                    num_lr = 0.001/10
            elif eval(FLAGS.better_care):
                if j == 0:
                    num_lr = eval(FLAGS.num_lr)
                    print('initial learning_rate: ', num_lr, '\n')
                else:
                    num_lr = num_lr/10
 
            else:    
                if j == 0:
                    num_lr = eval(FLAGS.num_lr)
                    print('initial learning_rate: ', num_lr, '\n')
                if j==5 or j==10:
                    num_lr = num_lr/10
 

            for i in range(int(np.ceil(50000 / 256))):

                image_batch, label_batch = sess.run([train_images, train_labels])

                _, loss_eval = sess.run([train_op, loss], feed_dict={x: image_batch, y: label_batch, is_training: False, lr: num_lr})
                if i == 0 and j == 0:
                    first_loss = loss_eval
                if i==0:
                    print('begin loss: ', loss_eval)
            print('end loss:', loss_eval)


            test_acc = 0
            test_count = 0
            time_count = 0
            
            for i in range(int(np.floor(10000 / 250))):

                image_batch, label_batch = sess.run([test_images, test_labels])
                
                start = time.time()
                acc = sess.run(accuracy, feed_dict={x: image_batch, y: label_batch, is_training: False})
                end = time.time()

                time_count += end-start
                test_acc += acc
                test_count += 1

            test_acc /= test_count
            time_ave_one_epoch = time_count/ 10 # /10000*1000=/10ms per image
            print(time_ave_one_epoch)

            if j>=epoch-3: 
                print(epoch)
                ave_time += time_ave_one_epoch/3
            
            print("Test Accuracy:  " + str(test_acc))


            f.write('\n\nepoch= ' + str(j) + '\n')
            f.write('learning_rate'+str(num_lr)+'\n')
            if j == 0:
                f.write('first loss: ' + str(first_loss) + '\n')
            f.write('final loss: ' + str(loss_eval) + '\n')
            f.write('accuracy_after_this_Train: ' + str(test_acc) + '\n')
            f.write('time_ave_one_epoch : ' + str(time_ave_one_epoch) + '\n\n')
            if j+1==epoch:
                f.write('\n\nave runtime: '+str( ave_time) + 'ms gpu per image\n\n')
                print(ave_time)
                saver.save(sess, ckpt_root_path+'/resnet8_pcsvd-acc'+str(test_acc)[0:7]+'-ave3time'+str(ave_time)[0:7], global_step=j+1)
            else:
                saver.save(sess, ckpt_root_path+'/resnet8_pcsvd-acc'+str(test_acc)[0:7]+'-time'+str(time_ave_one_epoch)[0:7], global_step=j+1)


            
        f.close()

    finally:
            coord.request_stop()
            coord.join(threads)

