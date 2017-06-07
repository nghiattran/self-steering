from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorkit.base import ObjectiveBase
import tensorflow as tf


class Objective(ObjectiveBase):
    def loss(self, hypes, logits, labels):
        labels = tf.reshape(labels, [-1, 1])
        error = tf.subtract(labels, logits['output'])
        rmse_loss = tf.sqrt(tf.reduce_mean(tf.square(error)))
        loss = rmse_loss

        reg_loss_col = tf.GraphKeys.REGULARIZATION_LOSSES
        weight_loss = tf.add_n(tf.get_collection(reg_loss_col), name='reg_loss')

        total_loss = weight_loss + loss

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('weight_loss', weight_loss)
        tf.summary.scalar('total_loss', total_loss)

        losses = {
            'total_loss': total_loss,
            'mse_loss': rmse_loss,
            'weight_loss': weight_loss,
            'error': tf.reduce_sum(tf.abs(error))
        }

        return losses

    def evaluate(self, hyp, images, target, logits, losses):
        eval_list = []
        eval_list.append(('Total loss', losses['total_loss']))
        eval_list.append(('RMSE loss', losses['mse_loss']))
        eval_list.append(('Error', losses['error']))
        eval_list.append(('Weights', losses['weight_loss']))
        return eval_list