import tensorflow as tf
import numpy as np
import os
import cifar10_input

import time




os.environ['CUDA_VISIBLE_DEVICES']='1'




'''Network'''
with tf.device('/cpu:0'):
    test_images, test_labels = cifar10_input.inputs(data_dir='/home/test01/csw/resnet18/cifar-10-batches-bin', eval_data=True, batch_size=250)




with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())



    coord = tf.train.Coordinator()
    threads = []
    for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
    print('coord done \n')

    saver = tf.train.import_meta_graph('/home/test01/csw/resnet8/ckpt_250epoch_another_bn_api/resnet8-acc0.93009-250.meta')
    graph = tf.get_default_graph()

    x = graph.get_tensor_by_name('x:0')
    y = graph.get_tensor_by_name('y:0')
    is_training = graph.get_tensor_by_name("is_training:0")
    accuracy = graph.get_tensor_by_name('accuracy:0')

    saver.restore(sess, '/home/test01/csw/resnet8/ckpt_250epoch_another_bn_api/resnet8-acc0.93009-250')

    log_path  = './resnet8_orig_time.log'

    try:
        f = open(log_path, 'a')
        epoch=10
        ave_time=0
        for j in range(epoch):

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
            f.write('accuracy_after_this_Train: ' + str(test_acc) + '\n')
            f.write('time_ave_one_epoch : ' + str(time_ave_one_epoch) + '\n\n')
            if j+1==epoch:
                f.write('\n\n\nave runtime: '+str( ave_time) + 'ms gpu per image\n\n')
                print(ave_time)

        f.close()
    finally:

        coord.request_stop()
        coord.join(threads)










