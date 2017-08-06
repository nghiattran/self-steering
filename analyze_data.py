from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import matplotlib.pyplot as plt

def read_csv(file):
    with open(file, 'r') as f:
        data = f.readlines()[1:]

    angles = []
    for i, entry in enumerate(data):
        frame_id, angle = entry.split(',')

        angles.append(float(angle))

    return np.array(angles)

def draw(data, name, BINS):
    mean = np.mean(data)
    std = np.std(data)

    plt.hist(data, BINS, alpha=0.3, label='%s, mean: %f, std: %f' % (name, mean, std))

if __name__ == '__main__':

    train_data = read_csv(os.path.join('DATA', 'train', 'train.csv'))
    val_data = read_csv(os.path.join('DATA', 'train', 'val.csv'))
    test_data = read_csv(os.path.join('DATA', 'test', 'test.csv'))

    maximum = max(np.max(train_data), np.max(val_data), np.max(test_data))
    minimum = min(np.min(train_data), np.min(val_data), np.min(test_data))

    print(maximum, minimum)

    BINS = np.linspace(-1, 1, 100)

    plt.xlabel('Steering angle')
    plt.ylabel('Frequency')

    draw(train_data, name='train', BINS=BINS)
    draw(val_data, name='val', BINS=BINS)
    draw(test_data, name='test', BINS=BINS)

    plt.legend(loc='upper right')

    plt.show()