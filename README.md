# self-steering
Using Deep Learning to Predict Steering Angles. [Udacity Challenge #2](https://medium.com/udacity/challenge-2-using-deep-learning-to-predict-steering-angles-f42004a36ff3)

## Installation

Install all dependencies with:

```bash
$ pip install -r requirements.txt
```

Also, make sure you have Tensorflow above 1.0.

## Usage

### 1. Training

You can train your own model by using.

```bash
$ python train.py --hypes path-to-your-hype
```

See [train.py]() for more detail.

Check [Hypes](#hypes) to see all options or to create your own ones.

### 2. TensorVision

This project is built on top of [TensorVision](http://tensorvision.readthedocs.io/en/master/) which helps organizing 
experiments.

There are times that training process is interupted due to unexpected reasons. In this case, you can resume training by 
using `tv-continue`.

```bash
$ tv-continue --logdir path-to-logdir
```

To evaluate trained model on valuation set, you can use `tv-analyze`.

```bash
$ tv-analyze --logdir path-to-logdir
```

### 3. Submission

To test your model, use `submission.py` to generate csv file. Format of this `csv` file would be similar
ro [CH2_final_evaluation.csv]().

```bash
$ python submission.py path-to-logdir path-image-folder
```

See [submission.py]() for more detail.

### 4. Visualization

Use `visualize.py` to create a demo video from `csv` files generated above. 

```bash
$ python visualize.py path-to-input-csv path-groundtruth-csv path-to-image-folder
```

See [visualize.py]() for more detail.

## Hypes

#### Options:

* `model`:
    * `input_file`: path to python file that handles input.
    * `architecture`: path to python file that constructs main graph.
    * `objective_file`: path to python file that handles losses.
    * `evaluator_file`: path to python file that handles avaluation.
* `data`:
    * `train_file`: path to `train.csv`
    * `val_file`: path to `val.csv`
* `logging`:
    * `display_iter`: display frequency.
    * `eval_iter`: evaluatoion frequency.
    * `write_iter`: writing summary for Tensorboard frequency.
    * `save_iter`: saving model frequency.
* `solver`:
    * `opt`: type of optimizer. Supported `Adam`, `RMS`, and `SGD`. 
    * `rnd_seed`: seed number for random.
    * `epsilon`: value for epsilon (a small number that is used to avoid zero division).
    * `learning_rate`
    * `max_steps`
* `clip_norm`: upper bound to avoid gradient exploding. 
* `image_height`
* `image_width`
* `batch_size`
* `reg_strength`: regularization strength.
* `color_space`: image preprocessing color space. Supported `rgb` and `yuv`. Default `rgb`.
* `crop`: crop size. This value count from bottom to top.

#### Usage

To fine-tune, you only need to change `learning_rate`, `opt`, and `max_steps` in `solver` and a few other
 hyperparameters like `reg_strength`, `color_space`, `crop`,...
 
You can also create your own neural network architect by using layout in `architecture/nvidia.py` and keep
 other parameter as is.
 
## Models

* `nvidia`: inspired by [SullyChen](https://github.com/SullyChen/Autopilot-TensorFlow)'s implementation of Nvidia's paper
 [End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316.pdf)

## Acknowledge

This project uses similar layout as in [MarvinTeichmann](https://github.com/MarvinTeichmann)'s 
[KittiBox](https://github.com/MarvinTeichmann/KittiBox). Especially, `generic_optimizer.py` and 
`udacity_input.py`.