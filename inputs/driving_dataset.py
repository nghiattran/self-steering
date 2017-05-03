import itertools
import random
import os
import tensorflow as tf
import threading
import pandas as pd
import numpy as np
import scipy as scp
import scipy.misc

def load_image(path, hypes):
    img = scp.misc.imread(path)
    img = scp.misc.imresize(img[-150:], size=(hypes["image_height"], hypes["image_width"])) / 255.0
    return img

def load_csv(csv_file):
    data = pd.read_csv(csv_file, names=['filename', 'angle'])
    return data

def load_data(interpolate_csv, hypes, jitter=False, random_shuffel=True):
    base_path = os.path.realpath(os.path.dirname(interpolate_csv))
    data = load_csv(interpolate_csv)

    files = data['filename'][1:].tolist()
    angles = data['angle'][1:].tolist()
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

            yield {"image": im, "angle": [float(angles[i])]}


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
        return image, angle
    else:
        assert("Bad phase: {}".format(phase))