# encoding=utf-8

import tensorflow as tf


def add_hidden_layer_summary(value, tag, detail=False):
    tf.summary.scalar("%s/fraction_of_zero_values" % tag,
                      tf.nn.zero_fraction(value))
    if detail:
        tf.summary.scalar("%s/max_value" % tag, tf.reduce_max(tf.reduce_max(value, axis=1), axis=0))
        tf.summary.scalar("%s/min_value" % tag, tf.reduce_min(tf.reduce_min(value, axis=1), axis=0))
        val_mean, val_variance = tf.nn.moments(value, axes=[1])
        tf.summary.scalar("%s/variance" % tag, tf.reduce_mean(val_variance, axis=0))


def add_net_abs_mean_summary(value, tag):
    net_abs_mean = tf.reduce_mean(tf.abs(value))
    tf.summary.scalar("abs_mean/%s" % tag, net_abs_mean)
    abs_value = tf.abs(value)
    greater_mask = tf.greater(abs_value, 1e-12)
    net_abs_greater_mean = tf.reduce_mean(tf.boolean_mask(abs_value, greater_mask))
    tf.summary.scalar("abs_greater_0_mean/%s" % tag, net_abs_greater_mean)


def add_net0_summary(net0_dict):
    for name, net0_tensor in net0_dict.items():
        add_hidden_layer_summary(net0_tensor, name)
        add_net_abs_mean_summary(net0_tensor, name)
        tf.summary.scalar("%s/max_value" % name, tf.reduce_max(net0_tensor))
        tf.summary.scalar("%s/min_value" % name, tf.reduce_min(net0_tensor))
        val_mean, val_variance = tf.nn.moments(net0_tensor, axes=[1])
        tf.summary.scalar("%s/variance" % name, tf.reduce_mean(val_variance))
