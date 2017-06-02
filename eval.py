from __future__ import print_function

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


BINS = np.linspace(-2, 2, 100)


def load_data(filepath):
    data = pd.read_csv(filepath, usecols=['frame_id', 'steering_angle'], index_col=None)
    data.sort_values('frame_id')
    files = data['frame_id'][1:].tolist()
    angles = data['steering_angle'][1:].tolist()
    return np.array(files), np.array(angles, dtype=np.float32)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Path viewer')
    parser.add_argument('input', type=str, help='Path to prediction file.')
    parser.add_argument('groundtargets', type=str, help='Path to groundtargets file.')
    args = parser.parse_args()

    pred_ids, preds = load_data(args.input)
    targets_ids, targets = load_data(args.groundtargets)

    min_shape = min(preds.shape[0], targets.shape[0])

    targets = targets[:min_shape]
    preds = preds[:min_shape]

    # Sanity check
    pred_ids = pred_ids[:min_shape]
    targets_ids = targets_ids[:min_shape]
    if np.sum(targets_ids - pred_ids) != 0:
        print(np.sum(targets_ids - pred_ids) )
        print('error')

    saved_path = 'REPORT'

    error = np.abs(targets - preds)
    rmse = np.mean(np.square(error)) ** 0.5

    plotfile = os.path.join(saved_path, 'targets_vs_predictions_histogram_step.png')
    plt.clf()
    plt.xlabel('Steering angle')
    plt.ylabel('Frequency')
    plt.hist(targets, BINS, alpha=0.5, label='targets')
    plt.hist(preds, BINS, alpha=0.5, label='predictions')
    plt.legend(loc='upper right')
    plt.savefig(plotfile)

    plotfile = os.path.join(saved_path, 'targets_vs_predictions_scatter_step.png')
    plt.clf()
    start = - np.pi
    end = np.pi
    plt.scatter(targets, preds, s=10)
    plt.xlabel('Targets')
    plt.ylabel('Predictions')
    plt.plot([start, end], [start, end], color='red')
    plt.savefig(plotfile)

    plotfile = os.path.join(saved_path, 'angles_vs_error_scatter_step.png')
    plt.clf()
    plt.scatter(targets, error, s=10)
    plt.xlabel('Angle')
    plt.ylabel('Errors')
    plt.savefig(plotfile)

    print('Sum error', abs(np.sum(error)))
    print('Max error:', np.max(error))
    print('Mean error:', np.mean(error))
    print('Min error:', np.min(error))
    print('Root-mean-square error:', rmse)