from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np

def read_data(filepath):
    with open(filepath, 'r') as f:
        data = f.readlines()[1:]

    for i, entry in enumerate(data):
        file, angle = entry.split(',')
        file = os.path.join('Ch2_002', 'center', file + '.jpg')
        data[i] = '%s,%s' % (file, angle)
    return data

if __name__ == '__main__':
    filepath = 'DATA/train/interpolated.csv'
    basepath = os.path.realpath(os.path.dirname(filepath))

    val_csv = os.path.join(basepath, 'val.csv')
    train_csv = os.path.join(basepath, 'train.csv')
    header = 'filename,steering_angle\n'

    data = read_data(filepath)
    random.shuffle(data)

    split = int(len(data) // 10)
    train_data = data[split:]
    val_data = data[:split]

    with open(train_csv, 'w') as f:
        f.write(header)
        f.write(''.join(train_data))

    with open(val_csv, 'w') as f:
        f.write(header)
        f.write(''.join(val_data))

    filepath = 'DATA/test/interpolated.csv'
    basepath = os.path.realpath(os.path.dirname(filepath))

    test_csv = os.path.join(basepath, 'test.csv')
    test_data = read_data(filepath)
    with open(test_csv, 'w') as f:
        f.write(header)
        f.write(''.join(test_data))