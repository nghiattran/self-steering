#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Nghia Tran

"""
Generate csv for predicting steering angles.

Usage:
usage: submission.py [-h] [--limit LIMIT] [--save SAVE] logdir test_folder

Create submission for Udacity.

positional arguments:
  logdir                Path to logdir.
  test_folder           Path to test folder.

optional arguments:
  -h, --help            show this help message and exit
  --limit LIMIT, -l LIMIT
                        Number of files.
  --save SAVE, -s SAVE  Save file.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import argparse
import os
import sys
import logging
from inputs.udacity_input import load_image


def load(logdir):
    import tensorflow as tf
    import tensorvision.utils as tv_utils
    import tensorvision.core as core

    tv_utils.set_gpus_to_use()

    # Loading hyperparameters from logdir
    hypes = tv_utils.load_hypes_from_logdir(logdir, base_path='hypes')

    logging.info("Hypes loaded successfully.")

    # Loading tv modules (encoder.py, decoder.py, eval.py) from logdir
    modules = tv_utils.load_modules_from_logdir(logdir)
    logging.info("Modules loaded successfully. Starting to build tf graph.")

    with tf.Graph().as_default():
        # Create placeholder for input
        image_pl = tf.placeholder(tf.float32, shape=(hypes["image_height"], hypes["image_width"], 3))
        image = tf.expand_dims(image_pl, 0)
        # build Tensorflow graph using the model from logdir
        prediction = core.build_inference_graph(hypes, modules,
                                                image=image)

        logging.info("Graph build successfully.")

        # Create a session for running Ops on the Graph.
        sess = tf.Session()
        saver = tf.train.Saver()

        # Load weights from logdir
        core.load_weights(logdir, sess, saver)

        logging.info("Weights loaded successfully.")

    return image_pl, prediction, sess, hypes


def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)

    parser = argparse.ArgumentParser(description='Create submission for Udacity.')
    parser.add_argument('logdir', type=str, help='Path to logdir.')
    parser.add_argument('test_folder', type=str, help='Path to test folder.')
    parser.add_argument('--limit', '-l', type=int, default=-1, help='Number of files.')
    parser.add_argument('--save', '-s', type=str, default='submission.csv', help='Save file.')

    args = parser.parse_args()
    logdir = args.logdir
    image_pl, prediction, sess, hypes = load(logdir)

    save_file = args.save
    files = sorted(os.listdir(args.test_folder))[:args.limit]

    if len(files) == 0:
        logging.warning('No image found at path %s' % args.test_folder)
        exit(1)

    start = time.time()
    with open(save_file, 'w') as f:
        f.write('frame_id,steering_angle\n')
        for i, file in enumerate(files):
            sys.stdout.write('\r>> Processubg %d/%d images' % (i + 1, len(files)))
            sys.stdout.flush()

            filepath = os.path.join(args.test_folder, file)

            img = load_image(path=filepath, hypes=hypes)
            feed = {image_pl: img}

            output = prediction['output']
            pred = sess.run(output,
                            feed_dict=feed)
            pred = pred[0][0]

            frame_id = os.path.splitext(file)[0]
            f.write('%s,%f\n' % (frame_id, pred))

    time_taken = time.time() - start
    logging.info('Video saved as %s' % save_file)
    logging.info('Number of images: %d' % len(files))
    logging.info('Time takes: %.2f s' % (time_taken))
    logging.info('Frequency: %.2f fps' % (len(files) / time_taken))


if __name__ == '__main__':
    main()