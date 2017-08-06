from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf

from data_utils import int64_feature, float_feature, bytes_feature

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert data to tfRecords.')
    parser.add_argument('input_path', type=str, help='Path to input csv file.')
    parser.add_argument('--output', '-o', type=str, default=os.path.join('TFRecords', 'udacity.tfrecord'),
                        help='Path to destination TFRecord file.')

    args = parser.parse_args()

    filepath = args.input_path
    tf_filename = args.output
    datadir = os.path.realpath(os.path.dirname(filepath))

    try:
        os.makedirs(os.path.dirname(tf_filename))
    except OSError as e:
        # Be happy with already created directory
        pass

    with open(filepath, 'r') as f:
        data = f.readlines()[1:]

    with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
        for i, entry in enumerate(data):
            sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, len(data)))
            sys.stdout.flush()

            filename, angle = entry.split(',')
            angle = float(angle)

            filepath = os.path.join(datadir, filename)
            image_data = tf.gfile.FastGFile(filepath, 'r').read()

            frame_id, _ = os.path.splitext(os.path.basename(filepath))
            frame_id = int(frame_id)

            features = tf.train.Features(feature={
                'steering_angle': float_feature(angle),
                'frame_id': int64_feature(frame_id),
                'image_raw': bytes_feature(image_data)
            })

            example = tf.train.Example(features=features)
            tfrecord_writer.write(example.SerializeToString())

    print('\nFinished converting to tfRecord!')