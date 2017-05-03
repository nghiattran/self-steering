from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os


cnt = 0

def conv_weight_variable(shape):
    global cnt
    cnt += 1
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    # initializer = tf.random_uniform_initializer(-0.1, 0.1)
    w = tf.get_variable('weight_%d' % (cnt), shape=shape, initializer=initializer)

    initializer = tf.constant(0.1, shape=[shape[-1]])
    b = tf.get_variable('bias_%d' % (cnt), initializer=initializer)
    return w, b

def weight_variable(shape):
    return conv_weight_variable(shape)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')

def inference(hypes, images, train=True):
    # Reset counter so that train and val network use same weights
    global cnt
    cnt = 0

    # If train, set keep_prob to keep_prob value in hype
    # If val, set to 1.0 which mean no dropout
    keep_prob = hypes.get('keep_prob', 1.0) if train else 1.0

    regularizer = tf.contrib.layers.l2_regularizer(hypes.get('regularizer_strength', 0.5))
    with tf.variable_scope('network', regularizer=regularizer):
        with tf.name_scope("conv1"):
            # first convolutional layer
            conv1_w, conv1_b = conv_weight_variable([5, 5, 3, 24])
            h_conv1 = tf.nn.relu(conv2d(images, conv1_w, 2) + conv1_b)

        with tf.name_scope("conv2"):
            # second convolutional layer
            conv2_w, conv2_b = conv_weight_variable([5, 5, 24, 36])
            h_conv2 = tf.nn.relu(conv2d(h_conv1, conv2_w, 2) + conv2_b)

        with tf.name_scope("conv3"):
            # third convolutional layer
            conv3_w, conv3_b = conv_weight_variable([5, 5, 36, 48])
            h_conv3 = tf.nn.relu(conv2d(h_conv2, conv3_w, 2) + conv3_b)

        with tf.name_scope("conv4"):
            # fourth convolutional layer
            conv4_w, conv4_b = conv_weight_variable([3, 3, 48, 64])
            h_conv4 = tf.nn.relu(conv2d(h_conv3, conv4_w, 1) + conv4_b)

        with tf.name_scope("conv5"):
            # fifth convolutional layer
            conv5_w, conv5_b = conv_weight_variable([3, 3, 64, 64])

            h_conv5 = tf.nn.relu(conv2d(h_conv4, conv5_w, 1) + conv5_b)

        with tf.name_scope("fc6"):
            # FCL 1
            fc6_w, fc6_b = weight_variable([1152, 1164])

            h_conv5_flat = tf.reshape(h_conv5, [-1, 1152])
            h_fc6 = tf.nn.relu(tf.matmul(h_conv5_flat, fc6_w) + fc6_b)
            h_fc6 = tf.contrib.layers.batch_norm(h_fc6, is_training=train, trainable=True)
            h_fc6_drop = tf.nn.dropout(h_fc6, keep_prob=keep_prob)

        with tf.name_scope("fc7"):
            # FCL 2
            fc7_w, f7_b = weight_variable([1164, 100])
            h_fc7 = tf.nn.relu(tf.matmul(h_fc6_drop, fc7_w) + f7_b)
            h_fc7 = tf.contrib.layers.batch_norm(h_fc7, is_training=train, trainable=True)
            h_fc7_drop = tf.nn.dropout(h_fc7, keep_prob=keep_prob)

        with tf.name_scope("fc8"):
            # FCL 3
            fc8_w, f8_b = weight_variable([100, 50])
            h_fc8 = tf.nn.relu(tf.matmul(h_fc7_drop, fc8_w) + f8_b)
            h_fc8 = tf.contrib.layers.batch_norm(h_fc8, is_training=train, trainable=True)
            h_fc8_drop = tf.nn.dropout(h_fc8, keep_prob=keep_prob)

        with tf.name_scope("fc9"):
            # FCL 3
            fc9_w, f9_b = weight_variable([50, 10])
            h_fc9 = tf.nn.relu(tf.matmul(h_fc8_drop, fc9_w) + f9_b)
            h_fc9_drop = tf.nn.dropout(h_fc9, keep_prob=keep_prob)

        with tf.name_scope('fc10'):
            # Output
            fc10_w, fc10_b = weight_variable([10, 1])
            output = tf.multiply(tf.atan(tf.matmul(h_fc9_drop, fc10_w) + fc10_b), 2)

    return {
        'output': output
    }