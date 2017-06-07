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

        self.cnt += 1
    
        return w, b


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')

def build_stream1(images, variable_handler, keep_prob):
    with tf.name_scope('Stream1'):
        with tf.name_scope("Conv1"):
            # first convolutional layer
            conv1_w, conv1_b = variable_handler.weight_variable([5, 5, 3, 24])
            h_conv1 = tf.nn.relu(conv2d(images, conv1_w, stride=2) + conv1_b)
    
        with tf.name_scope("Conv2"):
            # second convolutional layer
            conv2_w, conv2_b = variable_handler.weight_variable([5, 5, 24, 36])
            h_conv2 = tf.nn.relu(conv2d(h_conv1, conv2_w, stride=2) + conv2_b)
    
        with tf.name_scope("Conv3"):
            # third convolutional layer
            conv3_w, conv3_b = variable_handler.weight_variable([5, 5, 36, 48])
            h_conv3 = tf.nn.relu(conv2d(h_conv2, conv3_w, stride=2) + conv3_b)
    
        with tf.name_scope("Conv4"):
            # fourth convolutional layer
            conv4_w, conv4_b = variable_handler.weight_variable([3, 3, 48, 64])
            h_conv4 = tf.nn.relu(conv2d(h_conv3, conv4_w, stride=2) + conv4_b)
    
        with tf.name_scope("Conv5"):
            # fifth convolutional layer
            conv5_w, conv5_b = variable_handler.weight_variable([3, 3, 64, 64])
    
            h_conv5 = tf.nn.relu(conv2d(h_conv4, conv5_w, stride=2) + conv5_b)
    
        with tf.name_scope("Conv6"):
            # fifth convolutional layer
            conv6_w, conv6_b = variable_handler.weight_variable([3, 3, 64, 64])
    
            h_conv6 = tf.nn.relu(conv2d(h_conv5, conv6_w, stride=2) + conv6_b)
    
        with tf.name_scope("FC7"):
            h_conv6_flat = tf.reshape(h_conv6, [-1, 512])
    
            fc7_w, fc7_b = variable_handler.weight_variable([512, 100])
            h_fc7 = tf.nn.relu(tf.matmul(h_conv6_flat, fc7_w) + fc7_b)
    
            h_fc7_drop = tf.nn.dropout(h_fc7, keep_prob=keep_prob)
    
        with tf.name_scope("FC8"):
            fc8_w, f8_b = variable_handler.weight_variable([100, 50])
            h_fc8 = tf.nn.relu(tf.matmul(h_fc7_drop, fc8_w) + f8_b)
    
            h_fc8_drop = tf.nn.dropout(h_fc8, keep_prob=keep_prob)
    
        with tf.name_scope("FC9"):
            fc9_w, f9_b = variable_handler.weight_variable([50, 10])
            h_fc9 = tf.matmul(h_fc8_drop, fc9_w) + f9_b

            h_fc9_drop = tf.nn.dropout(h_fc9, keep_prob=keep_prob)

        with tf.name_scope("Output"):
            fc10_w, fc10_b = variable_handler.weight_variable([10, 1])
            h_fc10 = tf.matmul(h_fc9_drop, fc10_w) + fc10_b
    
    return h_fc10


def build_stream2(images, variable_handler, keep_prob):
    with tf.name_scope('Stream2'):
        with tf.name_scope("Conv1"):
            # first convolutional layer
            conv1_w, conv1_b = variable_handler.weight_variable([5, 5, 3, 24])
            h_conv1 = tf.nn.relu(conv2d(images, conv1_w, stride=2) + conv1_b)

        with tf.name_scope("Conv2"):
            # second convolutional layer
            conv2_w, conv2_b = variable_handler.weight_variable([5, 5, 24, 36])
            h_conv2 = tf.nn.relu(conv2d(h_conv1, conv2_w, stride=2) + conv2_b)

        with tf.name_scope("Conv3"):
            # third convolutional layer
            conv3_w, conv3_b = variable_handler.weight_variable([5, 5, 36, 48])
            h_conv3 = tf.nn.relu(conv2d(h_conv2, conv3_w, stride=2) + conv3_b)

        with tf.name_scope("Conv4"):
            # fourth convolutional layer
            conv4_w, conv4_b = variable_handler.weight_variable([3, 3, 48, 64])
            h_conv4 = tf.nn.relu(conv2d(h_conv3, conv4_w, stride=2) + conv4_b)

        with tf.name_scope("Conv5"):
            # fifth convolutional layer
            conv5_w, conv5_b = variable_handler.weight_variable([3, 3, 64, 64])

            h_conv5 = tf.nn.relu(conv2d(h_conv4, conv5_w, stride=2) + conv5_b)

        with tf.name_scope("FC6"):
            h_conv6_flat = tf.reshape(h_conv5, [-1, 1344])

            fc6_w, fc6_b = variable_handler.weight_variable([1344, 100])
            h_fc6 = tf.nn.relu(tf.matmul(h_conv6_flat, fc6_w) + fc6_b)

            h_fc6_drop = tf.nn.dropout(h_fc6, keep_prob=keep_prob)

        with tf.name_scope("FC7"):
            fc7_w, f7_b = variable_handler.weight_variable([100, 50])
            h_fc7 = tf.nn.relu(tf.matmul(h_fc6_drop, fc7_w) + f7_b)

            h_fc7_drop = tf.nn.dropout(h_fc7, keep_prob=keep_prob)

        with tf.name_scope("FC8"):
            fc8_w, f8_b = variable_handler.weight_variable([50, 10])
            h_fc8 = tf.matmul(h_fc7_drop, fc8_w) + f8_b

            h_fc8_drop = tf.nn.dropout(h_fc8, keep_prob=keep_prob)

        with tf.name_scope("Output"):
            fc9_w, fc9_b = variable_handler.weight_variable([10, 1])
            h_fc5 = tf.matmul(h_fc8_drop, fc9_w) + fc9_b

    return h_fc5


def build_stream3(images, variable_handler, keep_prob):
    with tf.name_scope('Stream3'):
        with tf.name_scope("Conv1"):
            # first convolutional layer
            conv1_w, conv1_b = variable_handler.weight_variable([8, 8, 3, 16])
            h_conv1 = tf.nn.relu(conv2d(images, conv1_w, stride=4) + conv1_b)

        with tf.name_scope("Conv2"):
            # second convolutional layer
            conv2_w, conv2_b = variable_handler.weight_variable([5, 5, 16, 32])
            h_conv2 = tf.nn.relu(conv2d(h_conv1, conv2_w, stride=2) + conv2_b)

        with tf.name_scope("Conv3"):
            # third convolutional layer
            conv3_w, conv3_b = variable_handler.weight_variable([5, 5, 32, 64])
            h_conv3 = tf.nn.relu(conv2d(h_conv2, conv3_w, stride=2) + conv3_b)

        with tf.name_scope("FC4"):
            h_conv3_flat = tf.reshape(h_conv3, [-1, 4160])

            fc4_w, fc4_b = variable_handler.weight_variable([4160, 512])
            h_fc4 = tf.matmul(h_conv3_flat, fc4_w) + fc4_b

            h_fc4_drop = tf.nn.dropout(h_fc4, keep_prob=keep_prob)

        with tf.name_scope("Output"):
            fc5_w, fc5_b = variable_handler.weight_variable([512, 1])
            h_fc5 = tf.matmul(h_fc4_drop, fc5_w) + fc5_b

    return h_fc5

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

            # rescale to [-1,1] instead of [0, 1)
            images = tf.subtract(images, 0.5)
            images = tf.multiply(images, 2.0)

        keep_prob = hypes.get('keep_prob', 1.0) if is_training else 1.0
        with tf.variable_scope('Network', regularizer=regularizer, initializer=initializer):
            stream1 = build_stream1(images=images,
                                    variable_handler=vh,
                                    keep_prob=keep_prob)

            stream2 = build_stream2(images=images,
                                    variable_handler=vh,
                                    keep_prob=keep_prob)

            stream3 = build_stream3(images=images,
                                    variable_handler=vh,
                                    keep_prob=keep_prob)
                
            output = stream1 + stream2 + stream3

        return {
            'output': output
        }