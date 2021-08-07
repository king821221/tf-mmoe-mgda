import tensorflow as tf


def gradient_normalize_func(grads, losses, normalization_type='l2'):
    gn = {}

    if normalization_type == 'l2':
        for t in grads.keys():
            gn[t] = tf.sqrt(tf.reduce_sum(
                [tf.reduce_sum(tf.pow(gr, 2)) for gr in grads[t]]))
    elif normalization_type == 'loss':
        for t in grads:
            gn[t] = losses[t]
    elif normalization_type == 'loss+':
        for t in grads:
            gn[t] = losses[t] * tf.sqrt(tf.reduce_sum(
                [tf.reduce_sum(tf.pow(gr, 2)) for gr in grads[t]]))
    elif normalization_type == 'none':
        for t in grads:
            gn[t] = 1.0
    else:
        raise KeyError('ERROR: Invalid Normalization Type {}'
                       .format(normalization_type))

    return gn