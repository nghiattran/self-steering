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


def run_test(hypes, image_pl, sess, output_node, eval_list, validation=True, limit=-1):
    val_path = make_val_dir(hypes, validation)
    image_list = []
    stage = 'Val' if validation else 'Train'

    if validation:
        val_csv = os.path.join(hypes['dirs']['data_dir'], hypes['data']['val_file'])
    else:
        val_csv = os.path.join(hypes['dirs']['data_dir'], hypes['data']['train_file'])

    base_path = os.path.realpath(os.path.dirname(val_csv))
    data = load_csv(val_csv)
    files = data['filename'][1:].tolist()
    targets = np.array(data['angle'][1:].tolist(), dtype=np.float32)
    if limit > 0:
        files = files[:limit]
        targets = targets[:limit]

    preds = []
    with open(os.path.join(val_path, 'interpolated.csv'), 'w') as f:
        f.write('frame_id,prediction,steering_angle, difference\n')

        for i in range(len(files)):
            img_path = os.path.join(base_path, files[i])
            img = load_image(img_path, hypes)

            feed = {image_pl: img}
            pred, = sess.run([output_node], feed_dict=feed)
            preds.append(pred[0][0])

            image_list.append((os.path.basename(files[i]), img))

            filename = os.path.basename(files[i])
            frame_id = os.path.splitext(filename)[0]
            f.write('%s,%f,%f,%f\n' % (frame_id, pred[0], targets[i], targets[i] - pred[0]))

    preds = np.array(preds, dtype=np.float32)

    error = targets - preds
    rmse = np.sqrt(np.mean(np.square(error)))

    eval_list.append(('%s   sum error' % stage, abs(np.sum(error))))
    eval_list.append(('%s   max error' % stage, np.max(error)))
    eval_list.append(('%s   mean error' % stage, np.mean(error)))
    eval_list.append(('%s   min error' % stage, np.min(error)))
    eval_list.append(('%s   root-mean-square error' % stage, rmse))
    eval_list.append(('%s   root-mean-square variance' % stage, np.var(error)))

    # Create graphs
    step = hypes.get('step', hypes['logging']['eval_iter'])

    plotfile = os.path.join(val_path, 'targets_vs_predictions_histogram_step_%d.png' % step)
    plt.clf()
    plt.xlabel('Steering angle')
    plt.ylabel('Frequency')
    plt.hist(targets, BINS, alpha=0.5, label='targets')
    plt.hist(preds, BINS, alpha=0.5, label='predictions')
    plt.legend(loc='upper right')
    plt.savefig(plotfile)

    plotfile = os.path.join(val_path, 'targets_vs_predictions_scatter_step_%d.png' % step)
    plt.clf()
    start = np.min(np.minimum(targets, preds))
    end = np.max(np.maximum(targets, preds))
    plt.scatter(targets, preds)
    plt.xlabel('Targets')
    plt.ylabel('Predictions')
    plt.plot([start, end], [start, end])
    plt.savefig(plotfile)

    plotfile = os.path.join(val_path, 'predictions_vs_error_scatter_step_%d.png' % step)
    plt.clf()
    plt.scatter(preds, error)
    plt.xlabel('Predictions')
    plt.ylabel('Errors')
    plt.savefig(plotfile)

    return eval_list, image_list, feed


def evaluate(hypes, sess, image_pl, logits):
    eval_list = []
    output_node = logits['output']

    # Run test on valuation set
    eval_list, image_list, _ = run_test(hypes=hypes,
                                     sess=sess,
                                     image_pl=image_pl,
                                     output_node=output_node,
                                     eval_list=eval_list)

    # Run test on valuation set
    eval_list, image_list, feed = run_test(hypes=hypes,
                                           sess=sess,
                                           image_pl=image_pl,
                                           output_node=output_node,
                                           eval_list=eval_list,
                                           validation=False,
                                           limit=len(image_list))

    step = hypes.get('step', hypes['logging']['eval_iter'])
    hypes['step'] = step + hypes['logging']['eval_iter']

    start_time = time.time()
    for i in xrange(100):
        sess.run([output_node], feed_dict=feed)
    dt = (time.time() - start_time) / 100

    eval_list.append(('Speed (msec)', 1000 * dt))
    eval_list.append(('Speed (fps)', 1 / dt))

    return eval_list, image_list