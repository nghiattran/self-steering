from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import pandas as pd
import scipy as scp
import numpy as np
import matplotlib.pyplot as plt


BINS = np.linspace(-2, 2, 100)

def load_image(path, hypes):
    img = scp.misc.imread(path)
    img = scp.misc.imresize(img, size=(hypes["image_height"], hypes["image_width"]))
    return img

def load_csv(csv_file):
    data = pd.read_csv(csv_file, names=['filename', 'angle'])
    return data

def make_val_dir(hypes, validation=True):
    if validation:
        val_dir = os.path.join(hypes['dirs']['output_dir'], 'val_out')
    else:
        val_dir = os.path.join(hypes['dirs']['output_dir'], 'train_out')
    if not os.path.exists(val_dir):
        os.mkdir(val_dir)
    return val_dir

def evaluate(hypes, sess, image_pl, logits):
    val_path = make_val_dir(hypes)
    eval_list = []
    image_list = []

    output_node = logits['output']
    val_csv = os.path.join(hypes['dirs']['data_dir'], hypes['data']['val_file'])
    base_path = os.path.realpath(os.path.dirname(val_csv))

    data = load_csv(val_csv)
    files = data['filename'][1:].tolist()
    preds = []

    with open(os.path.join(val_path, 'interpolated.csv'), 'w') as f:
        f.write('frame_id,steering_angle\n')

        for i in range(len(files)):
            img_path = os.path.join(base_path, files[i])
            img = load_image(img_path, hypes)

            feed = {image_pl: img}
            pred, = sess.run([output_node], feed_dict=feed)
            preds.append(pred[0][0])

            image_list.append((os.path.basename(files[i]), img))

            filename = os.path.basename(files[i])
            frame_id = os.path.splitext(filename)[0]
            f.write('%s,%f\n' % (frame_id, pred[0]))

    start_time = time.time()
    for i in xrange(100):
        sess.run([output_node], feed_dict=feed)
    dt = (time.time() - start_time) / 100

    targets = np.array(data['angle'][1:].tolist(), dtype=np.float32)
    preds = np.array(preds, dtype=np.float32)

    error = targets - preds
    rmse = np.sqrt(np.mean(np.square(error)))


    eval_list.append(('Sum error', abs(np.sum(error))))
    eval_list.append(('Root-mean-square error', rmse))
    eval_list.append(('Root-mean-square variance', np.var(error)))
    eval_list.append(('Speed (msec)', 1000 * dt))
    eval_list.append(('Speed (fps)', 1 / dt))

    step = hypes.get('step', hypes['logging']['eval_iter'])

    plotfile = os.path.join(val_path, 'histogram_step_%d.png' % step)
    plt.clf()
    plt.xlabel('Steering angle')
    plt.ylabel('Frequency')
    plt.hist(targets, BINS, alpha=0.5, label='target')
    plt.hist(preds, BINS, alpha=0.5, label='pred')
    plt.legend(loc='upper right')
    plt.savefig(plotfile)

    plotfile = os.path.join(val_path, 'scatter_step_%d.png' % step)
    plt.clf()
    start = np.min(np.minimum(targets, preds))
    end = np.max(np.maximum(targets, preds))
    plt.scatter(targets, preds)
    plt.xlabel('Targets')
    plt.ylabel('Predictions')
    plt.plot([start, end], [start, end])
    plt.savefig(plotfile)

    hypes['step'] = step + hypes['logging']['eval_iter']

    return eval_list, image_list