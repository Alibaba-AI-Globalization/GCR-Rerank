import tensorflow as tf


def compare_same(tensor, n=10):
    """
      :param tensor: [BatchSize, N]
      :param n:
      :return: [B, N, N]
      """
    t1 = tf.expand_dims(tensor, -1)
    t2 = tf.expand_dims(tensor, 1)
    same = tf.cast(tf.equal(t1, t2), dtype=tf.int8)
    eye = tf.eye(n, dtype=tf.int8)
    same = same - eye
    return same


def float_context_statistic(tensor, item_num=10):
    """
      :param tensor: [B, N]
      :param item_num:
      :return: [B, N]
      """
    mean = tf.reduce_mean(tensor, axis=1, keep_dims=True)
    square = tf.square(tensor - mean)
    std = tf.sqrt(tf.reduce_sum(square, axis=1, keep_dims=True) / (tf.cast((item_num - 1), tf.float32) + 1e-9))
    mean = tf.tile(mean, [1, item_num])
    std = tf.tile(std, [1, item_num])
    return mean, std


def calc_bigger(tensor):
    """
      :param tensor: [B, N]
      :return: [B, N, N]
      """
    t1 = tf.expand_dims(tensor, -1)
    t2 = tf.expand_dims(tensor, 1)
    cmp = t1 - t2
    zeros = tf.zeros_like(cmp)
    one = tf.ones_like(cmp)
    biggerRes = tf.where(cmp >= 0, x=zeros, y=one)
    return biggerRes


def argsort(bigger):
    """
      :param bigger: calcBigger result [B, N, N]
      :return: [B, N]
      """
    sort_res = tf.reduce_sum(bigger, axis=2)
    sort_res = (sort_res - 4.5) / 3.0
    return sort_res


def calc_bias(tensor, mean, std):
    bias = (tensor - mean) / (std + 1e-8)
    bias = tf.where(tf.is_nan(bias), tf.zeros_like(bias), bias)
    return bias


def pad_sparse2dense(st, default_value="", pad_len=1):
    pad = tf.SparseTensor(indices=[[0, 0]], values=[default_value], dense_shape=[1, pad_len])
    st_concat = tf.sparse_concat(axis=1, sp_inputs=[st, pad], expand_nonconcat_dim=True)
    st_dense = tf.sparse_tensor_to_dense(st_concat, default_value=default_value)
    st_dense = tf.slice(st_dense, [0, 0], [-1, pad_len])
    return st_dense
