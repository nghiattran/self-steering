from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorkit.base import ArchitectBase
import tensorflow as tf

summary = {}


class VariableHandler():
    def __init__(self):
        self.cnt = 0

    def weight_variable(self, shape):
        weight_name = 'weight_%d' % (self.cnt)
        bias_name = 'bias_%d' % (self.cnt)

        w = tf.get_variable(name=weight_name,
                            shape=shape)

        initializer = tf.constant(0.1, shape=[shape[-1]])
        b = tf.get_variable(name=bias_name,
                            initializer=initializer)

        if bias_name not in summary:
            summary[bias_name] = b
            summary[weight_name] = w

            tf.summary.scalar('%s_sparsity' % weight_name, tf.nn.zero_fraction(w))
            tf.summary.scalar('%s_sparsity' % bias_name, tf.nn.zero_fraction(b))

        self.cnt += 1

        return w, b


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')


def deconv2d(x, W, output_shape, stride):
    return tf.nn.conv2d_transpose(x, W, output_shape=output_shape, strides=[1, stride, stride, 1], padding='VALID')


class Architect(ArchitectBase):
    def build_graph(self, hypes, input, phase):
        vh = VariableHandler()

        is_training = phase == 'train'

        initializer_type = hypes.get('initializer', 'xavier')
        if initializer_type == 'truncated_normal':
            initializer = tf.truncated_normal_initializer(stddev=0.1)
        elif initializer_type == 'random_uniform':
            initializer = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
        else:
            initializer = tf.contrib.layers.xavier_initializer_conv2d()

        reg_strength = hypes.get('reg_strength', 0.1)
        regularizer = None
        if reg_strength > 0:
            regularizer = tf.contrib.layers.l2_regularizer(reg_strength)

        with tf.name_scope('Preprocess'):
            input = tf.cast(input, tf.float32)
            images = tf.reshape(input, [-1, hypes['image_height'], hypes['image_width'], 3])

            with tf.name_scope('normalization'):
                # rescale to [-1,1] instead of [0, 1)
                images = tf.subtract(images, 0.5)
                images = tf.multiply(images, 2.0)

        with tf.variable_scope('Network', regularizer=regularizer, initializer=initializer):
            # If train, set keep_prob to keep_prob value in hype
            # If val, set to 1.0 which mean no dropout
            keep_prob = hypes.get('keep_prob', 1.0) if is_training else 1.0

            with tf.name_scope("conv1"):
                conv1_w, conv1_b = vh.weight_variable([5, 5, 3, 24])
                h_conv1 = tf.nn.relu(conv2d(images, conv1_w, 2) + conv1_b)

                tf.summary.image(
                    name='input',
                    tensor=[images[0]],
                    max_outputs=1
                )

                h, w, c = h_conv1[0].get_shape().as_list()
                feature_space = tf.reshape(h_conv1[0], shape=(c, h, w, 1))
                tf.summary.image(
                    name='fist_layer_activation',
                    tensor=feature_space,
                    max_outputs=c
                )

            with tf.name_scope("conv2"):
                conv2_w, conv2_b = vh.weight_variable([5, 5, 24, 36])
                h_conv2 = tf.nn.relu(conv2d(h_conv1, conv2_w, 2) + conv2_b)

            with tf.name_scope("conv3"):
                conv3_w, conv3_b = vh.weight_variable([5, 5, 36, 48])
                h_conv3 = tf.nn.relu(conv2d(h_conv2, conv3_w, 2) + conv3_b)

            with tf.name_scope("conv4"):
                conv4_w, conv4_b = vh.weight_variable([3, 3, 48, 64])
                h_conv4 = tf.nn.relu(conv2d(h_conv3, conv4_w, 1) + conv4_b)

            with tf.name_scope("conv5"):
                conv5_w, conv5_b = vh.weight_variable([3, 3, 64, 64])

                h_conv5 = tf.nn.relu(conv2d(h_conv4, conv5_w, 1) + conv5_b)

            with tf.name_scope("fc6"):
                fc6_w, fc6_b = vh.weight_variable([1152, 1164])

                h_conv5_flat = tf.reshape(h_conv5, [-1, 1152])
                h_fc6 = tf.nn.relu(tf.matmul(h_conv5_flat, fc6_w) + fc6_b)

                h_fc6_drop = tf.nn.dropout(h_fc6, keep_prob=keep_prob)

            with tf.name_scope("fc7"):
                fc7_w, f7_b = vh.weight_variable([1164, 100])
                h_fc7 = tf.nn.relu(tf.matmul(h_fc6_drop, fc7_w) + f7_b)

                h_fc7_drop = tf.nn.dropout(h_fc7, keep_prob=keep_prob)

            with tf.name_scope("fc8"):
                fc8_w, f8_b = vh.weight_variable([100, 50])
                h_fc8 = tf.nn.relu(tf.matmul(h_fc7_drop, fc8_w) + f8_b)

                h_fc8_drop = tf.nn.dropout(h_fc8, keep_prob=keep_prob)

            with tf.name_scope("fc9"):
                fc9_w, f9_b = vh.weight_variable([50, 10])
                h_fc9 = tf.nn.relu(tf.matmul(h_fc8_drop, fc9_w) + f9_b)

                h_fc9_drop = tf.nn.dropout(h_fc9, keep_prob=keep_prob)

            with tf.name_scope('fc10'):
                fc10_w, fc10_b = vh.weight_variable([10, 1])

                output = tf.multiply(tf.atan(tf.matmul(h_fc9_drop, fc10_w) + fc10_b), 2, name='y')

        return {
            'output': output
        }