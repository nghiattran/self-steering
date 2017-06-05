import itertools
import random
import os

import skimage
import tensorflow as tf
import threading
import pandas as pd
import numpy as np
import scipy as scp
import scipy.misc
from skimage.color import rgb2yuv


# This function inspired by
# https://github.com/MarvinTeichmann/KittiBox/blob/405b29c3ce8936e8e39d098cb5f27df026459d88/inputs/kitti_input.py
def _processe_image(hypes, image):
    image = tf.cast(image, tf.float32)

    augment_level = hypes.get('augment_level', -1)
    if augment_level == 0:
        image = tf.image.random_brightness(image, max_delta=30)
        image = tf.image.random_contrast(image, lower=0.75, upper=1.25)
    elif augment_level == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.6)
        image = tf.image.random_hue(image, max_delta=0.15)
    elif augment_level == 2:
        image = skimage.util.random_noise(image, mode='gaussian')
    elif augment_level == 3:
        image = skimage.util.random_noise(image, mode='speckle')

    return image

def load_csv(csv_file):
    data = pd.read_csv(file, names=['filename', 'angle'])
    return data

def load_data(file, hypes, jitter=False, random_shuffel=True):
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([file])
    _, serialized_example = reader.read(filename_queue)

    for _ in itertools.count():
        features = tf.parse_single_example(
            serialized_example,
            features={
                'steering_angle': tf.FixedLenFeature([], tf.float32),
                'frame_id': tf.FixedLenFeature([], tf.int64),
                'image_raw': tf.FixedLenFeature([], tf.string)
            }
        )
        img = tf.decode_raw(features['image_raw'], tf.uint8)
        img.set_shape((480 * 640 * 3))
        img = tf.reshape(img, (480, 640, 3))

        crop = hypes.get('crop', 400)
        if crop > 0:
            img = img[:crop]

        img = tf.image.resize_images(img, size=(hypes["image_height"], hypes["image_width"]))

        yield {"image": img, "angle": [features['steering_angle']]}


def start_enqueuing_threads(hypes, q, phase, sess):
    """Start enqueuing threads."""

    # Creating Placeholder for the Queue
    x_in = tf.placeholder(tf.float32)
    angle_in = tf.placeholder(tf.float32)

    # Creating Enqueue OP
    enqueue_op = q.enqueue((x_in, angle_in))

    def make_feed(data):
        return {x_in: data['image'],
                angle_in: data['angle']}

    def thread_loop(sess, enqueue_op, gen):
        for d in gen:
            sess.run(enqueue_op, feed_dict=make_feed(d))

    data_file = hypes["data"]['%s_file' % phase]
    data_dir = hypes['dirs']['data_dir']
    data_file = os.path.join(data_dir, data_file)

    gen = load_data(data_file, hypes,
                    jitter={'train': hypes['solver']['use_jitter'],
                            'val': False}[phase])

    data = gen.next()
    sess.run(enqueue_op, feed_dict=make_feed(data))
    t = threading.Thread(target=thread_loop,
                         args=(sess, enqueue_op, gen))
    t.daemon = True
    t.start()

def create_queues(hypes, phase):
    """Create Queues."""
    dtypes = [tf.float32, tf.float32]
    shapes = ([hypes['image_height'], hypes['image_width'], 3],
              [1])
    capacity = hypes['batch_size'] * 10
    q = tf.FIFOQueue(capacity=capacity, dtypes=dtypes, shapes=shapes)
    return q

def inputs(hypes, q, phase):
    if phase == 'val':
        image, angle = q.dequeue()
        return image, angle
    elif phase == 'train':
        image, angle = q.dequeue_many(hypes['batch_size'])
        image = _processe_image(hypes=hypes, image=image)
        return image, angle
    else:
        assert("Bad phase: {}".format(phase))