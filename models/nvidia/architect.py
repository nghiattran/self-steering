from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorkit.base import ArchitectBase
import tensorflow as tf

cnt = 0
summary = {}

def weight_variable(shape):
    global cnt
    w = tf.get_variable('weight_%d' % (cnt), shape=shape)

    initializer = tf.constant(0.1, shape=[shape[-1]])
    b = tf.get_variable('bias_%d' % (cnt), initializer=initializer)

    if 'bias_%d' % (cnt) not in summary:
        summary['bias_%d' % (cnt)] = b
        summary['weight_%d' % (cnt)] = w

    cnt += 1

    return w, b


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')


def deconv2d(x, W, output_shape, stride):
    return tf.nn.conv2d_transpose(x, W, output_shape=output_shape, strides=[1, stride, stride, 1], padding='VALID')


class Architect(ArchitectBase):
    def build_graph(self, hypes, input, phase):
        # Reset counter so that train and val networks use same weights
        global cnt
        cnt = 0

        initializer_type = hypes.get('initializer', 'xavier')
        if initializer_type == 'truncated_normal':
            initializer = tf.truncated_normal_initializer(stddev=0.1)
        elif initializer_type == 'random_uniform':
            initializer = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
        else:
            initializer = tf.contrib.layers.xavier_initializer_conv2d()

        regularizer = tf.contrib.layers.l2_regularizer(hypes.get('reg_strength', 0.1))

        with tf.name_scope('Preprocess'):
            input = tf.cast(input, tf.float32)
            images = tf.reshape(input, [-1, hypes['image_height'], hypes['image_width'], 3])

            # rescale to [-1,1] instead of [0, 1)
            images = tf.subtract(images, 0.5)
            images = tf.multiply(images, 2.0)

        with tf.variable_scope('network', regularizer=regularizer, initializer=initializer):
            # If train, set keep_prob to keep_prob value in hype
            # If val, set to 1.0 which mean no dropout
            keep_prob = hypes.get('keep_prob', 1.0) if phase == 'train' else 1.0

            with tf.name_scope("conv1"):
                # first convolutional layer
                conv1_w, conv1_b = weight_variable([5, 5, 3, 24])
                h_conv1 = tf.nn.relu(conv2d(images, conv1_w, 2) + conv1_b)

            with tf.name_scope("conv2"):
                # second convolutional layer
                conv2_w, conv2_b = weight_variable([5, 5, 24, 36])
                h_conv2 = tf.nn.relu(conv2d(h_conv1, conv2_w, 2) + conv2_b)

            with tf.name_scope("conv3"):
                # third convolutional layer
                conv3_w, conv3_b = weight_variable([5, 5, 36, 48])
                h_conv3 = tf.nn.relu(conv2d(h_conv2, conv3_w, 2) + conv3_b)

            with tf.name_scope("conv4"):
                # fourth convolutional layer
                conv4_w, conv4_b = weight_variable([3, 3, 48, 64])
                h_conv4 = tf.nn.relu(conv2d(h_conv3, conv4_w, 1) + conv4_b)

            with tf.name_scope("conv5"):
                # fifth convolutional layer
                conv5_w, conv5_b = weight_variable([3, 3, 64, 64])

                h_conv5 = tf.nn.relu(conv2d(h_conv4, conv5_w, 1) + conv5_b)

            with tf.name_scope("fc6"):
                # FCL 1
                fc6_w, fc6_b = weight_variable([1152, 1164])

                h_conv5_flat = tf.reshape(h_conv5, [-1, 1152])
                h_fc6 = tf.nn.relu(tf.matmul(h_conv5_flat, fc6_w) + fc6_b)
                h_fc6_drop = tf.nn.dropout(h_fc6, keep_prob=keep_prob)

            with tf.name_scope("fc7"):
                # FCL 2
                fc7_w, f7_b = weight_variable([1164, 100])
                h_fc7 = tf.nn.relu(tf.matmul(h_fc6_drop, fc7_w) + f7_b)
                h_fc7_drop = tf.nn.dropout(h_fc7, keep_prob=keep_prob)

            with tf.name_scope("fc8"):
                # FCL 3
                fc8_w, f8_b = weight_variable([100, 50])
                h_fc8 = tf.nn.relu(tf.matmul(h_fc7_drop, fc8_w) + f8_b)
                h_fc8_drop = tf.nn.dropout(h_fc8, keep_prob=keep_prob)

            with tf.name_scope("fc9"):
                # FCL 3
                fc9_w, f9_b = weight_variable([50, 10])
                h_fc9 = tf.nn.relu(tf.matmul(h_fc8_drop, fc9_w) + f9_b)
                h_fc9_drop = tf.nn.dropout(h_fc9, keep_prob=keep_prob)

            with tf.name_scope('fc10'):
                # Output
                fc10_w, fc10_b = weight_variable([10, 1])

                output = tf.matmul(h_fc9_drop, fc10_w) + fc10_b

        return {
            'output': output
        }