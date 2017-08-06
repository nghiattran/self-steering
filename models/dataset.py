from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops.image_ops_impl import ResizeMethod, _Check3DImage, fix_image_flip_shape
from tensorkit.base import DatasetBase, DatasetsBase
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import array_ops


def decode_jpeg(image, scope=None):
  """Decode a JPEG string into one 3-D float image Tensor.
  Args:
    image: scalar string Tensor.
    scope: Optional scope for name_scope.
  Returns:
    3-D float Tensor with values ranging from [0, 1).
  """
  with tf.name_scope(values=[image], name=scope,
                     default_name='decode_jpeg'):
    # Decode the string as an RGB JPEG.
    # Note that the resulting image contains an unknown height and width
    # that is set dynamically by decode_jpeg. In other words, the height
    # and width of image is unknown at compile-time.
    image = tf.image.decode_jpeg(image, channels=3)

    # After this point, all image pixels reside in [0,1)
    # until the very end, when they're rescaled to (-1, 1).  The various
    # adjust_* ops all require this range for dtype float.
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image


def parse_example_proto(serialized_example, hypes):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'steering_angle': tf.FixedLenFeature([], tf.float32),
            'frame_id': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string)
        }
    )

    image = decode_jpeg(features['image_raw'])
    image.set_shape(shape=(480, 640, 3))
    label = tf.cast(features['steering_angle'], tf.float32)
    frame_id = tf.cast(features['frame_id'], tf.int64)

    crop = hypes.get('crop', 400)
    if crop > 0:
        image = image[-crop:]

    return image, label, frame_id

def resize(image, thread_id, hypes, is_train):
    resize_method = thread_id % 4 if is_train else ResizeMethod.BICUBIC
    
    image = tf.image.resize_images(image,
                                   size=(hypes['image_height'], hypes['image_width']),
                                   method=resize_method)
    return image

def distort_image(image, thread_id, hypes):
    with tf.name_scope('Distort_image'):
        augment_level = hypes.get('augment_level', -1)
        if augment_level == 0:
            image = tf.image.random_brightness(image, max_delta=30)
            image = tf.image.random_contrast(image, lower=0.75, upper=1.25)
        elif augment_level == 1:
            image = tf.image.random_saturation(image, lower=0.5, upper=1.6)
            image = tf.image.random_hue(image, max_delta=0.15)
        elif augment_level == 2:
            color_ordering = thread_id % 3

            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            elif color_ordering == 1:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)

        image = tf.clip_by_value(image, 0.0, 1.0)

    return image

def image_preprocessing(image, label, thread_id, hypes, is_train, seed=None):
    image = resize(image, thread_id, hypes, is_train)

    if is_train:
        image = distort_image(image, thread_id, hypes)

        with tf.name_scope('Random_flip'):
            # Check image's dimension and generate random variable
            image = ops.convert_to_tensor(image, name='image')
            _Check3DImage(image, require_static=False)
            uniform_random = random_ops.random_uniform([], 0, 1.0, seed=seed)
            mirror_cond = math_ops.less(uniform_random, .5)

            # Flip image
            image_result = control_flow_ops.cond(mirror_cond,
                                                 lambda: array_ops.reverse(image, [1]),
                                                 lambda: image)
            image = fix_image_flip_shape(image, image_result)

            # Flip steering angle
            label_result = control_flow_ops.cond(mirror_cond,
                                                 lambda: -label,
                                                 lambda: label)
            result_shape = label.get_shape()
            label_result.set_shape(result_shape)
            label = label_result

    return image, label


class Dataset(DatasetBase):
    def __init__(self, record_file, hypes, is_train):
        self.hypes = hypes

        # Count number of sample in record file
        cnt = 0
        for _ in tf.python_io.tf_record_iterator(record_file):
            cnt += 1
        self._length = cnt

        with tf.device('/cpu:0'):
            phase = 'Train_reader' if is_train else 'Inference_reader'
            with tf.name_scope(phase):
                filename_queue = tf.train.string_input_producer([record_file])
                reader = tf.TFRecordReader()
                _, serialized_example = reader.read(filename_queue)

                batch_size = self.hypes['solver']['batch_size']
                num_threads = self.hypes['solver'].get('num_threads', 1) if is_train else 1

                assert num_threads > 0, 'Number of threads need to be bigger than 0.'

                images_and_labels = []
                for thread_id in range(num_threads):
                    with tf.name_scope('Thread_process_%d' % thread_id):
                        # Parse a serialized Example proto to extract the image and metadata.
                        image, label, frame_id = parse_example_proto(serialized_example, hypes)
                        image, label = image_preprocessing(image=image,
                                                           label=label,
                                                           thread_id=thread_id,
                                                           hypes=hypes,
                                                           is_train=is_train)
                        images_and_labels.append([image, label])

                    if thread_id == 0:
                        self.image = image
                        self.label = label
                        self.frame_id = frame_id

                memory_factor = hypes.get('memory_factor', 2) if is_train else 2

                self.images, self.labels = tf.train.shuffle_batch_join(
                    images_and_labels,
                    batch_size=batch_size,
                    min_after_dequeue=num_threads * batch_size,
                    capacity=memory_factor * num_threads * batch_size)


    def get_frame_id(self, index):
        return self._files[index]

    def next_batch(self, batch_size):
        sess = tf.get_default_session()
        return sess.run([self.images, self.labels])

    def next_batch_num(self, batch_size):
        sess = tf.get_default_session()
        images = []
        angles = []
        frame_ids = []
        for i in range(batch_size):
            image, angle, frame_id = sess.run([self.image, self.label, self.frame_id])
            images.append(image)
            angles.append(angle)
            frame_ids.append(frame_id)

        return images, angles, frame_ids

    def __len__(self):
        return self._length


class Datasets(DatasetsBase):
    def create(self, hypes):
        train_dataset = Dataset(hypes['data']['train_file'], hypes, is_train=True)
        val_dataset = Dataset(hypes['data']['val_file'], hypes, is_train=False)
        self.set_datasets(
            train=train_dataset,
            validation=val_dataset
        )