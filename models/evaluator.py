from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorkit.base import EvaluatorBase
import os
import time
import numpy as np
import matplotlib.pyplot as plt


BINS = np.linspace(-3, 3, 100)
POS_BINS = np.linspace(0, 3, 20)

def make_val_dir(hypes, validation=True):
    if validation:
        val_dir = os.path.join(hypes['dirs']['log_dir'], 'val_out')
    else:
        val_dir = os.path.join(hypes['dirs']['log_dir'], 'train_out')
    if not os.path.exists(val_dir):
        os.mkdir(val_dir)
    return val_dir


def run_test(hypes, image_pl, sess, output_node, eval_list, dataset, validation=True, limit=-1, shuffle=False):
    val_path = make_val_dir(hypes, validation)
    stage = 'Val' if validation else 'Train'

    targets = []
    preds = []

    step = hypes.get('step', hypes['logging']['eval_iter'])

    with open(os.path.join(val_path, 'interpolated_step_%d.csv' % step), 'w') as f:
        f.write('frame_id,prediction,steering_angle, difference\n')
        for i in range(limit):
            img, angle, frame_id = dataset.next_batch_num(1)
            angle = angle[0]

            feed = {image_pl: img}
            pred, = sess.run([output_node], feed_dict=feed)

            pred_angle = pred[0][0]

            preds.append(pred_angle)
            targets.append(angle)

            f.write('%s,%f,%f,%f\n' % (frame_id[0], pred[0][0], targets[i], targets[i] - pred[0][0]))

    preds = np.array(preds, dtype=np.float32)

    error = np.abs(targets - preds)
    rmse = np.mean(np.square(error)) ** 0.5

    eval_list.append(('%s   sum error' % stage, abs(np.sum(error))))
    eval_list.append(('%s   max error' % stage, np.max(error)))
    eval_list.append(('%s   mean error' % stage, np.mean(error)))
    eval_list.append(('%s   min error' % stage, np.min(error)))
    eval_list.append(('%s   root-mean-square error' % stage, rmse))

    plotfile = os.path.join(val_path, 'targets_vs_predictions_histogram_step_%d.png' % step)
    plt.clf()
    plt.title('Step %d' % step, loc='left')
    plt.xlabel('Steering angle')
    plt.ylabel('Frequency')
    plt.hist(targets, BINS, alpha=0.5, label='targets')
    plt.hist(preds, BINS, alpha=0.5, label='predictions')
    plt.legend(loc='upper right')
    plt.savefig(plotfile)

    plotfile = os.path.join(val_path, 'targets_vs_predictions_scatter_step_%d.png' % step)
    plt.clf()
    plt.title('Step %d' % step, loc='left')
    start = - np.pi
    end = np.pi
    plt.scatter(targets, preds, s=10)
    plt.xlabel('Targets')
    plt.ylabel('Predictions')
    plt.plot([start, end], [start, end], color='red')
    plt.savefig(plotfile)

    plotfile = os.path.join(val_path, 'error_histogram_step_%d.png' % step)
    plt.clf()
    plt.title('Step %d' % step, loc='left')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.hist(error, POS_BINS, label='error')
    plt.savefig(plotfile)

    return eval_list, feed

class Evaluator(EvaluatorBase):
    def evaluate(self, hypes, sess, input_node, logits, datasets):
        eval_list = []
        output_node = logits['output']

        limit = hypes['solver']['batch_size'] if hypes['data'].get('truncated', False) else len(datasets.validation)
        shuffle = False if hypes['data'].get('truncated', False) else True

        # Run test on valuation set
        res = run_test(hypes=hypes,
                       sess=sess,
                       image_pl=input_node,
                       output_node=output_node,
                       eval_list=eval_list,
                       limit=limit,
                       shuffle=shuffle,
                       dataset=datasets.validation)
        eval_list, _ = res

        # Run test on training set
        res = run_test(hypes=hypes,
                       sess=sess,
                       image_pl=input_node,
                       output_node=output_node,
                       eval_list=eval_list,
                       validation=False,
                       limit=limit,
                       shuffle=shuffle,
                       dataset=datasets.train)
        eval_list, feed = res

        step = hypes.get('step', hypes['logging']['eval_iter'])
        hypes['step'] = step + hypes['logging']['eval_iter']

        start_time = time.time()
        for i in xrange(100):
            sess.run([output_node], feed_dict=feed)
        dt = (time.time() - start_time) / 100

        eval_list.append(('Speed (msec)', 1000 * dt))
        eval_list.append(('Speed (fps)', 1 / dt))

        return eval_list