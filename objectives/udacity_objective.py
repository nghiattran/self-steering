import tensorflow as tf


def decoder(hyp, logits, train):
    batch_size = hyp['batch_size']
    hyp['solver']['batch_size'] = batch_size
    if not train:
        hyp['batch_size'] = 1
    hyp['batch_size'] = batch_size
    return logits


def loss(hypes, logits, target):
    error = tf.subtract(logits['output'], target)
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
        'rmse_loss': rmse_loss,
        'weight_loss': weight_loss,
        'error': tf.abs(tf.reduce_sum(error))
    }

    return losses


def evaluation(hyp, images, target, logits, losses, global_step):
    eval_list = []
    eval_list.append(('Total loss', losses['total_loss']))
    eval_list.append(('RMSE loss', losses['rmse_loss']))
    eval_list.append(('Error', losses['error']))
    eval_list.append(('Weights', losses['weight_loss']))
    return eval_list