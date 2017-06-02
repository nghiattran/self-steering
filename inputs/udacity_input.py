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


def load_image(path, hypes, cb=None):
    img = scp.misc.imread(path)

    if callable(cb):
        cb(img, hypes)

    crop = hypes.get('crop', 400)
    if crop > 0:
        img = img[-crop:]

    img = scp.misc.imresize(img, size=(hypes["image_height"], hypes["image_width"]))

    return img


def load_csv(csv_file):
    data = pd.read_csv(csv_file, names=['filename', 'angle'])
    return data


def load_data(interpolate_csv, hypes, jitter=False, random_shuffel=True):
    base_path = os.path.realpath(os.path.dirname(interpolate_csv))
    data = load_csv(interpolate_csv)

    files = data['filename'][1:].tolist()
    angles = data['angle'][1:].tolist()

    if hypes['data'].get("truncated", False):
        files = files[:hypes['batch_size']]
        angles = np.array(angles[:hypes['batch_size']], np.float32)


    for epoch in itertools.count():
        if random_shuffel:
            c = list(zip(files, angles))
            random.shuffle(c)
            files, angles = zip(*c)
        for i, image_file in enumerate(files):
            image_file = os.path.join(base_path, image_file)
            assert os.path.exists(image_file), \
                "File does not exist: %s" % image_file

            im = load_image(image_file, hypes=hypes)
            angle = float(angles[i])

            # jitter_flip = np.random.random_integers(0, 1)
            # if jitter_flip == 1:
            #     im = np.fliplr(im)
            #     angle = -angle

            yield {"image": im, "angle": [angle]}


def start_enqueuing_threads(hypes, q, phase, sess):
    """Start enqueuing threads."""
    print('hi\n\n\n\n')
    # Creating Placeholder for the Queue
    x_in = tf.placeholder(tf.float32)
    angle_in = tf.placeholder(tf.float32)

    preprocesses = hypes.get('preprocesses', [])
    for p in preprocesses:
        if p == 'batchnorm':
            pass
        else:
            raise ValueError('Preprocess type %s is unsupported' % (str(p)))

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
        images, angle = q.dequeue()
        return images, angle
    elif phase == 'train':
        images, angles = q.dequeue_many(hypes['batch_size'])
        images = tf.map_fn(lambda img: _processe_image(hypes=hypes, image=img), images)

        return images, angles
    else:
        assert("Bad phase: {}".format(phase))