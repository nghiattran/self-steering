from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

import scipy

if __name__ == '__main__':
    filepath = 'DATA/driving_dataset/data.txt'
    basepath = os.path.realpath(os.path.dirname(filepath))
    val_csv = os.path.join(basepath, 'val.csv')
    train_csv = os.path.join(basepath, 'train.csv')
    header = 'filename,steering_angle\n'
    with open(filepath, 'r') as f:
        data = f.readlines()

    for i, entry in enumerate(data):
        file, angle = entry.split(' ')
        angle = float(angle) * scipy.pi / 180
        data[i] = '%s,%f\n' % (file, angle)

    split = int(len(data) // 10)
    train_data = data[split:]
    val_data = data[:split]

    with open(train_csv, 'w') as f:
        f.write(header)
        f.write(''.join(train_data))

    with open(val_csv, 'w') as f:
        f.write(header)
        f.write(''.join(val_data))