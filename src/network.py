import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

def UnPooling2x2ZeroFilled(x):
    # https://github.com/tensorflow/tensorflow/issues/2169
    out = tf.concat([x, tf.zeros_like(x)], 3)
    out = tf.concat([out, tf.zeros_like(out)], 2)

    sh = x.get_shape().as_list()
    if None not in sh[1:]:
        out_size = [-1, sh[1] * 2, sh[2] * 2, sh[3]]
        return tf.reshape(out, out_size)
    else:
        shv = tf.shape(x)
        ret = tf.reshape(out, tf.stack([-1, shv[1] * 2, shv[2] * 2, sh[3]]))
        ret.set_shape([None, None, None, sh[3]])
        return ret

def vgg_cae(inputs, is_training=True, learning_rate=0.01, weight_decay=0.0005):
    batch_size = tf.shape(inputs)[0]
    input_size = inputs.get_shape()[1].value
    assert(input_size == 227)
    assert(input_size == inputs.get_shape()[2].value)
    h1_size = 55
    h2_size = 27
    h3_size = 27
    h4_size = 13
    
    num_input = inputs.get_shape()[3].value
    num_h1 = 96
    f_size1 = 11
    stride1 = 4
    
    pool_size1 = 3
    pool_stride1 = 2
    
    num_h2 = 256
    f_size2 = 5
    stride2 = 1
    
    pool_size2 = 3
    pool_stride2 = 2
    
    initializer = tf.contrib.layers.variance_scaling_initializer()
    regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)
    
    weights = {
        'encoder_h1': tf.get_variable(name='encoder_h1', dtype=tf.float32, \
                                  shape = [f_size1, f_size1, num_input, num_h1], \
                                  initializer = initializer, regularizer=regularizer),
        'encoder_h2': tf.get_variable(name='encoder_h2', dtype=tf.float32, \
                                  shape = [f_size2, f_size2, num_h1, num_h2], \
                                  initializer = initializer, regularizer=regularizer)
    }
    init1 = tf.constant_initializer(np.zeros(num_h1))
    init2 = tf.constant_initializer(np.zeros(num_h2))
    init3 = tf.constant_initializer(np.zeros(num_input))
    biases = {
        'encoder_b1': tf.get_variable(name='encoder_b1', dtype=tf.float32, shape = [num_h1], initializer = init1),
        'encoder_b2': tf.get_variable(name='encoder_b2', dtype=tf.float32, shape = [num_h2], initializer = init2),
        'decoder_b2': tf.get_variable(name='decoder_b2', dtype=tf.float32, shape = [num_h1], initializer = init1),
        'decoder_b1': tf.get_variable(name='decoder_b1', dtype=tf.float32, shape = [num_input], initializer = init3)
    }
    
    conv1 = tf.nn.relu(tf.nn.bias_add( \
                                         tf.nn.conv2d(inputs, weights['encoder_h1'], \
                                                      [1, stride1, stride1, 1], padding='VALID'), biases['encoder_b1']))
    assert(h1_size==conv1.get_shape()[2].value)
    
    
    pool1 = tf.nn.max_pool(conv1, ksize=[1, pool_size1, pool_size1, 1], strides=[1, pool_stride1, pool_stride1, 1], padding='VALID')
    assert(h2_size==pool1.get_shape()[2].value)
    
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(pool1, weights['encoder_h2'], \
                                                      [1, stride2, stride2, 1], padding='SAME'), biases['encoder_b2']))
    assert(h3_size==conv2.get_shape()[2].value)
    
    # pool2 = tf.nn.max_pool(conv2, ksize=[1, pool_size2, pool_size2, 1], strides=[1, pool_stride2, pool_stride2, 1], padding='VALID')
    # assert(h4_size==pool2.get_shape()[2].value)
    
    z = conv2
    
    
    # unpool2 = tf.pad(UnPooling2x2ZeroFilled(pool2), [[0,0],[0,1],[0,1],[0,0]], "CONSTANT")
    # print(h3_size, unpool2.get_shape()[2].value)
    # assert(h3_size==unpool2.get_shape()[2].value)
    
    deconv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d_transpose(conv2, weights['encoder_h2'], \
                                                              tf.stack([batch_size, h2_size, h2_size, num_h1]), \
                                                              [1, stride2, stride2, 1], padding='SAME'),biases['decoder_b2']))
    print(h2_size, deconv2.get_shape()[2].value)
    # assert(h2_size==deconv2.get_shape()[2].value)
    
    unpool1 = tf.pad(UnPooling2x2ZeroFilled(deconv2), [[0,0],[0,1],[0,1],[0,0]], "CONSTANT")
    # assert(h1_size==unpool1.get_shape()[2].value)
    
    deconv1 = tf.nn.sigmoid(tf.nn.bias_add( \
                                           tf.nn.conv2d_transpose(unpool1, weights['encoder_h1'], \
                                                         tf.stack([batch_size, input_size, input_size, num_input]), \
                                                         [1, stride1, stride1, 1], padding='VALID'),biases['decoder_b1']))
    
    # assert(input_size==deconv1.get_shape()[2].value)
    y = deconv1-0.5
    
    recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - inputs),[1,2,3]))
    reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    total_loss = recon_loss + reg_loss
    
    # opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    grads = opt.compute_gradients(total_loss)
    apply_gradient_op = opt.apply_gradients(grads)
    
    tf.summary.scalar('losses/Recon_Loss', recon_loss)
    tf.summary.scalar('losses/Reg_Loss', reg_loss)
    tf.summary.scalar('losses/Total_Loss', total_loss)
    merged = tf.summary.merge_all()
    
    return y, total_loss, z, merged, apply_gradient_op
    