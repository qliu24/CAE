from datetime import datetime
import network
import time
import sys
import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils import *

start_step = 5000
total_step = 7000
learning_rate = 0.00001
print_step = 10
val_step = 100
val_iteration = 10
summary_step = 100
snapshot_step = 100

batch_size = 128
image_size = 224
stride = 2

# g_data_folder = '/export/home/qliu24/qing_voting_139/CAE/dataset/cifar-10-batches-py'
# data_file = os.path.join(g_data_folder, 'data_batch_{0}')
dloader = Data_loader_imagenet('./imagenet_file_list.txt')

g_cache_folder = '/export/home/qliu24/qing_voting_139/CAE/cache/'
checkpoints_dir = os.path.join(g_cache_folder, 'checkpoints')
log_folder = os.path.join(g_cache_folder, 'log')

input_images = tf.placeholder(tf.float32, [batch_size, image_size, image_size, 3])
with tf.variable_scope('vgg_cae'):
    y, total_loss, z, merged, apply_gradient_op = network.vgg_cae(input_images, stride = stride, learning_rate = learning_rate)

restorer = tf.train.Saver([var for var in tf.global_variables()])
saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
init_op = tf.global_variables_initializer()

print('init done')

with tf.Session(config=config) as sess:
    print(str(datetime.now()) + ': Start Init')
    sess.run(init_op)
    if start_step != 0:
        restorer.restore(sess, os.path.join(checkpoints_dir, 'fine_tuned-' + str(start_step)))
    
    print(str(datetime.now()) + ': Finish Init')
    
    train_writer = tf.summary.FileWriter(log_folder, graph=sess.graph)
    step = start_step + 1
    tf_time = 0
    while step <= total_step:
        batch_cur = dloader.get_batch(batch_size)
        
        time_before_tf = time.time()
        [out_loss, summary, _] = sess.run([total_loss, merged, apply_gradient_op], feed_dict={input_images: batch_cur})
        time_after_tf = time.time()
        tf_time += time_after_tf - time_before_tf

        if step % print_step == 0:
            print('{}: Iter {}: Training Loss = {:.5f}'.format(datetime.now(), step, out_loss))
            print('    where TF took {:.2f} seconds'.format(tf_time))
            tf_time = 0
        
        if step % summary_step == 0:
            train_writer.add_summary(summary, step)
        
        if step % snapshot_step == 0:
            saver.save(sess, os.path.join(checkpoints_dir, 'fine_tuned'), global_step=step)
        
        if step % val_step == 0:
            out_loss = []
            for vi in range(val_iteration):
                batch_eva = dloader.get_batch(batch_size,tp='eval')
                out_loss_i = sess.run(total_loss, feed_dict={input_images: batch_eva})
                out_loss.append(out_loss_i)
                
            print('{}: Iter {}: Evaluation Loss = {:.5f}'.format(datetime.now(), step, np.mean(out_loss)))
        
        step += 1
            
