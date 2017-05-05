'''
Results video generator Udacity Challenge 2
Original By: Comma.ai
Revd 1: Chris Gundling
Revd 2: Nghia Tran

Copy taken from https://github.com/udacity/self-driving-car/blob/master/steering-models/community-models/cg23/epoch_viewer.py
then revised by Nghia Tran

usage: visualize.py [-h] [--limit LIMIT] [--save SAVE]
                    input groundtruth basepath

Path viewer

positional arguments:
  input                 Path to prediction file.
  groundtruth           Path to groundtruth file.
  basepath              Path image folder.

optional arguments:
  -h, --help            show this help message and exit
  --limit LIMIT, -l LIMIT
                        Number of files.
  --save SAVE, -s SAVE  Save file.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import time
import scipy
import sys
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import pylab
import imageio
import matplotlib.patches as mpatches

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)

# ***** get perspective transform for images *****
from skimage import transform as tf

rsrc = \
    [[43.45456230828867, 118.00743250075844],
     [104.5055617352614, 69.46865203761757],
     [114.86050156739812, 60.83953551083698],
     [129.74572757609468, 50.48459567870026],
     [132.98164627363735, 46.38576532847949],
     [301.0336906326895, 98.16046448916306],
     [238.25686790036065, 62.56535881619311],
     [227.2547443287154, 56.30924933427718],
     [209.13359962247614, 46.817221154818526],
     [203.9561297064078, 43.5813024572758]]
rdst = \
    [[10.822125594094452, 1.42189132706374],
     [21.177065426231174, 1.5297552836484982],
     [25.275895776451954, 1.42189132706374],
     [36.062291434927694, 1.6376192402332563],
     [40.376849698318004, 1.42189132706374],
     [11.900765159942026, -2.1376192402332563],
     [22.25570499207874, -2.1376192402332563],
     [26.785991168638553, -2.029755283648498],
     [37.033067044190524, -2.029755283648498],
     [41.67121717733509, -2.029755283648498]]

tform3_img = tf.ProjectiveTransform()
tform3_img.estimate(np.array(rdst), np.array(rsrc))


def perspective_tform(x, y):
    p1, p2 = tform3_img((x, y))[0]
    return int(p2), int(p1)


# ***** functions to draw lines *****
def draw_pt(img, x, y, color, sz=1):
    # These offsets are needed to draw curves correctly for 4800x640 images. Added by Nghia Tran
    x_offset = 250
    y_offset = 182

    row, col = perspective_tform(x, y)
    if row >= 0 and row < img.shape[0] and \
            col >= 0 and col < img.shape[1]:
        img[row - sz + x_offset:row + sz + x_offset, col - sz + y_offset:col + sz + y_offset] = color


def draw_path(img, path_x, path_y, color):
    for x, y in zip(path_x, path_y):
        draw_pt(img, x, y, color)


# ***** functions to draw predicted path *****
def calc_curvature(v_ego, angle_steers, angle_offset=0):
    deg_to_rad = np.pi / 180.
    slip_fator = 0.0014  # slip factor obtained from real data
    steer_ratio = 15.3  # from http://www.edmunds.com/acura/ilx/2016/road-test-specs/
    wheel_base = 2.67  # from http://www.edmunds.com/acura/ilx/2016/sedan/features-specs/

    angle_steers_rad = (angle_steers - angle_offset)  # * deg_to_rad
    curvature = angle_steers_rad / (steer_ratio * wheel_base * (1. + slip_fator * v_ego ** 2))
    return curvature


def calc_lookahead_offset(v_ego, angle_steers, d_lookahead, angle_offset=0):
    # *** this function returns the lateral offset given the steering angle, speed and the lookahead distance
    curvature = calc_curvature(v_ego, angle_steers, angle_offset)

    # clip is to avoid arcsin NaNs due to too sharp turns
    y_actual = d_lookahead * np.tan(np.arcsin(np.clip(d_lookahead * curvature, -0.999, 0.999)) / 2.)
    return y_actual, curvature


def draw_path_on(img, speed_ms, angle_steers, color=(0, 0, 255)):
    path_x = np.arange(0., 50.1, 0.5)
    path_y, _ = calc_lookahead_offset(speed_ms, angle_steers, path_x)
    draw_path(img, path_x, path_y, color)


def load_data(filepath):
    data = pd.read_csv(filepath, usecols=['frame_id', 'steering_angle'], index_col=None)
    files = data['frame_id'][1:].tolist()
    angles = data['steering_angle'][1:].tolist()
    return files, np.array(angles, dtype=np.float32)


# ***** main loop *****
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Path viewer')
    parser.add_argument('input', type=str, help='Path to prediction file.')
    parser.add_argument('groundtruth', type=str, help='Path to groundtruth file.')
    parser.add_argument('basepath', type=str, help='Path image folder.')
    parser.add_argument('--limit', '-l', type=int, default=-1, help='Number of files.')
    parser.add_argument('--save', '-s', type=str, default='demo.mp4', help='Save file.')
    args = parser.parse_args()

    pred_ids, preds = load_data(args.input)
    truth_ids, truth = load_data(args.groundtruth)

    if args.limit > 0:
        pred_ids = pred_ids[:args.limit]
        preds = preds[:args.limit]

    # Save groundtruth imformation in a dict because pred_ids and truth_ids can mismatch
    truth_dict = dict(zip(truth_ids, truth))

    # Create second screen with matplotlib
    fig = pylab.figure(figsize=[6.4, 1.6], dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    # ax.legend(loc='upper left',fontsize=8)
    line1, = ax.plot([], [], 'b.-', label='Human')
    line2, = ax.plot([], [], 'r.-', label='Model')
    A = []
    B = []
    ax.legend(loc='upper left', fontsize=8)

    red = (255, 0, 0)
    blue = (0, 0, 255)
    # font = ImageFont.truetype("sans-serif.ttf", 16)

    speed_ms = 5
    save_file = args.save
    start = time.time()

    human_legend = mpatches.Patch(color='red', label='Human')
    model_legend = mpatches.Patch(color='blue', label='Model')
    pylab.legend(handles=[human_legend, model_legend])

    # Run through all images
    with imageio.get_writer(save_file, mode='I', fps=24) as writer:
        for i, frame_id in enumerate(pred_ids):
            filepath = os.path.join(args.basepath, str(frame_id) + '.jpg')
            img = scipy.misc.imread(filepath)

            predicted_steers = preds[i]

            if frame_id not in truth_dict:
                logging.error('Can\'t find frame_id %d in %s.' % (frame_id, args.groundtruth))
                exit(1)

            actual_steers = truth_dict[frame_id]

            draw_path_on(img, speed_ms, actual_steers / 5.0, color=blue)
            draw_path_on(img, speed_ms, predicted_steers / 5.0, color=red)

            im = Image.fromarray(np.uint8(img))
            draw = ImageDraw.Draw(im)
            draw.text((0, 0), 'Human Steering Angle: %f' % actual_steers, blue)
            draw.text((0, 0), 'Model Steering Angle: %f' % predicted_steers, red)

            # canvas.draw()
            # width, height = fig.get_size_inches() * fig.get_dpi()
            # img = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

            writer.append_data(np.array(img) )

    time_taken = time.time() - start
    logging.info('Video saved as %s' % save_file)
    logging.info('Number of images: %d' % len(pred_ids))
    logging.info('Time takes: %.2f s' % (time_taken))
    logging.info('Frequency: %.2f fps' % (len(pred_ids) / time_taken))
