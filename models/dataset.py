from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorkit.base import DatasetBase, DatasetsBase
import tensorflow as tf


def decode_jpeg(image_buffer, scope=None):
  """Decode a JPEG string into one 3-D float image Tensor.
  Args:
    image_buffer: scalar string Tensor.
    scope: Optional scope for name_scope.
  Returns:
    3-D float Tensor with values ranging from [0, 1).
  """
  with tf.name_scope(values=[image_buffer], name=scope,
                     default_name='decode_jpeg'):
    # Decode the string as an RGB JPEG.
    # Note that the resulting image contains an unknown height and width
    # that is set dynamically by decode_jpeg. In other words, the height
    # and width of image is unknown at compile-time.
    image = tf.image.decode_jpeg(image_buffer, channels=3)

    # After this point, all image pixels reside in [0,1)
    # until the very end, when they're rescaled to (-1, 1).  The various
    # adjust_* ops all require this range for dtype float.
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image


class Dataset(DatasetBase):
    def __init__(self, record_file, hypes, is_train):
        self.hypes = hypes
        self._cursor = 0

        cnt = 0
        for _ in tf.python_io.tf_record_iterator(record_file):
            cnt += 1
        self._length = cnt

        phase = 'Train_reader' if is_train else 'Inference_read'
        with tf.name_scope(phase):
            filename_queue = tf.train.string_input_producer([record_file])
            reader = tf.TFRecordReader()
            _, serialized_example = reader.read(filename_queue)

            features = tf.parse_single_example(
                serialized_example,
                features={
                    'steering_angle': tf.FixedLenFeature([], tf.float32),
                    'frame_id': tf.FixedLenFeature([], tf.int64),
                    'image_raw': tf.FixedLenFeature([], tf.string)
                }
            )

            image_tensor = decode_jpeg(features['image_raw'])
            image_tensor.set_shape(shape=(480, 640, 3))
            crop = hypes.get('crop', 400)
            if crop > 0:
                image_tensor =image_tensor[-crop:]
                image_tensor = tf.image.resize_images(image_tensor,
                                                      size=(hypes['image_height'], hypes['image_width']))

            self.image = image_tensor
            self.label = tf.cast(features['steering_angle'], tf.float32)
            self.frame_id = tf.cast(features['steering_angle'], tf.int64)


            distored_image = self.image
            if is_train:
                with tf.name_scope('Distort_image'):
                    augment_level = hypes.get('augment_level', -1)
                    if augment_level == 0:
                        distored_image = tf.image.random_brightness(distored_image, max_delta=30)
                        distored_image = tf.image.random_contrast(distored_image, lower=0.75, upper=1.25)
                    elif augment_level == 1:
                        distored_image = tf.image.random_saturation(distored_image, lower=0.5, upper=1.6)
                        distored_image = tf.image.random_hue(distored_image, max_delta=0.15)
                    elif augment_level == 2:
                        distored_image = tf.image.random_brightness(distored_image, max_delta=32. / 255.)
                        distored_image = tf.image.random_saturation(distored_image, lower=0.5, upper=1.5)
                        distored_image = tf.image.random_hue(distored_image, max_delta=0.2)
                        distored_image = tf.image.random_contrast(distored_image, lower=0.5, upper=1.5)

                    distored_image = tf.clip_by_value(distored_image, 0.0, 1.0)

            num_threads = 6
            batch_size = self.hypes['solver']['batch_size']
            self.images, self.labels = tf.train.shuffle_batch([distored_image, self.label],
                                                              batch_size=batch_size,
                                                              capacity= 2 * num_threads * batch_size,
                                                              num_threads=num_threads,
                                                              min_after_dequeue=num_threads * batch_size)

        self.shuffle()


    def shuffle(self):
        pass
        # sess = tf.get_default_session()
        # _, _ = sess.run([images, labels])

    def reset_cursor(self):
        self._cursor = 0

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