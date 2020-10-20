import tensorflow as tf
import numpy as np
import imagenet_data
import image_processing
import os
import time

os.environ['CUDA_VISIBLE_DEVICES']='0'



imagenet_data_val = imagenet_data.ImagenetData('validation')
val_images, val_labels = image_processing.inputs(imagenet_data_val, batch_size=500, num_preprocess_threads=16)



with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())



    

    saver = tf.train.import_meta_graph('/home/test01/csw/resnet34/ckpt3_20200617/raw_from_pythorch_0.65872/resnet34-Pytorch.meta')
    graph = tf.get_default_graph()
    # print({v.op.name for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)})

    # is_training = tf.placeholder(tf.bool,name='is_training')
    # x =  tf.placeholder(tf.float32, [None,224,224,3],name='x')
    # y =  tf.placeholder(tf.int64, [None],name='y')
    # lr = tf.placeholder(tf.float32, name='learning_rate')

    is_training = sess.graph.get_tensor_by_name("Placeholder:0")
    print(is_training)
    x = sess.graph.get_tensor_by_name("Placeholder_1:0")
    print(x)
    y = sess.graph.get_tensor_by_name("Placeholder_2:0")
    print(y)
    out_put = graph.get_tensor_by_name("softmax:0")
    accuracy = graph.get_tensor_by_name("accuracy:0")

    coord = tf.train.Coordinator()
    threads = []
    for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

    saver.restore(sess, '/home/test01/csw/resnet34/ckpt3_20200617/resnet-34-accTensor("accuracy:0", shape=(), dtype=float32)-5')  


    log_path  = './resnet34_orig_time.log'



    try:
        ave_time = 0
        epoch=4
        for j in range(epoch):
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

                f.write('\nepoch= '+ str(j)+'\n') 
                f.write('accuracy_after_this_Train: ' + str(test_acc)+'\n\n')
                f.write('time_ave_one_epoch : ' + str(time_ave_one_epoch) + '\n\n')
                   

                if j+1==epoch:
                    f.write('time_ave_all_epoch : ' + str(ave_time) + 'ms gpu per imag\n\n')




    finally:

        coord.request_stop()
        coord.join(threads)





