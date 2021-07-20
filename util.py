import tensorflow as tf

class Constants(object):
    TF_PRINT_FLAG = 0

def tf_print(tensor, message, summarize=100, level=0):
    if level>Constants.TF_PRINT_FLAG:
        tensor = tf.Print(tensor,
                          [tf.shape(tensor),
                           tf.reduce_min(tensor),
                           tf.reduce_max(tensor),
                           tf.reduce_mean(tensor),
                           tensor],
                          message=message,
                          summarize=summarize)
    return tensor