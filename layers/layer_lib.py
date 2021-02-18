# coding=utf-8
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
from util import tf_summary
from tensorflow.python.framework import ops


def self_attention(x):
    # x : [B, N, size]
    w = tf.matmul(x, tf.transpose(x, [0, 2, 1]))
    w = tf.nn.softmax(w, dim=2)
    res = tf.matmul(w, x)
    return res


def dot_attention(x, q):
    # x: [B, N, size], q: [B, size]
    q = tf.expand_dims(q, axis=1)
    w = tf.matmul(x, tf.transpose(q, [0, 2, 1]))
    w = tf.nn.softmax(w, dim=1)
    res = tf.reduce_sum(tf.multiply(w, x), axis=1)
    return res


def build_dnn_net(input_net, name="dnn", hidden_units=None, activation=None, dropout=0., input_bn=False, l2=0.0,
                  reused=False, is_training=True):
    print("build net with: name={}, hidden_units={}, activation={}, dropout={}, inputBn={}, l2={}, reused={}".format(
        name, hidden_units, activation, dropout, input_bn, l2, reused
    ))
    
    net = input_net
    dnn_parent_scope = name
    activation_func = []
    for act in activation:
        activation_func.append(get_activation(act))
    
    for layer_id, num_hidden_units in enumerate(hidden_units):
        hiddenlayer_scope = name + "/hidden_layer_%d" % layer_id
        
        with tf.variable_scope(hiddenlayer_scope,
                               values=(net,), reuse=reused) as dnn_hidden_layer_scope:
            # if is train and input_bn, then. add batch_norm
            if input_bn:
                net = tf.contrib.layers.batch_norm(net, center=True, scale=True, is_training=is_training,
                                                   scope='%s/%s/bn' % (dnn_parent_scope, hiddenlayer_scope),
                                                   epsilon=1e-9,
                                                   reuse=reused)
            # add regular if needed.
            if l2 > 1e-6:
                # add l2 regular
                l2regular = tf.contrib.layers.l2_regularizer(l2)
                net = tf.contrib.layers.fully_connected(
                    net,
                    num_hidden_units,
                    activation_fn=activation_func[layer_id],
                    variables_collections=[dnn_parent_scope],
                    weights_initializer=initializers.variance_scaling_initializer(),
                    scope=dnn_hidden_layer_scope,
                    weights_regularizer=l2regular,
                    biases_regularizer=l2regular,
                    reuse=reused)
            else:
                net = tf.contrib.layers.fully_connected(
                    net,
                    num_hidden_units,
                    activation_fn=activation_func[layer_id],
                    variables_collections=[dnn_parent_scope],
                    weights_initializer=initializers.variance_scaling_initializer(),
                    scope=dnn_hidden_layer_scope,
                    reuse=reused)
            if is_training and dropout is not None and dropout > 1e-6:
                net = tf.contrib.layers.dropout(
                    net,
                    keep_prob=(1.0 - dropout))
        tf_summary.add_hidden_layer_summary(net, hiddenlayer_scope)
        tf_summary.add_net_abs_mean_summary(net, hiddenlayer_scope)
    
    return net


def swish(_x, scope=None):
    with tf.variable_scope(name_or_scope=scope, default_name="swish"):
        _alpha = tf.get_variable("swish", shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(1),
                                 collections=[ops.GraphKeys.MODEL_VARIABLES, ops.GraphKeys.GLOBAL_VARIABLES])
        return _x * tf.nn.sigmoid(_alpha * _x)


def get_activation(name):
    act = {
        'relu': tf.nn.relu,
        'leaky_relu': tf.nn.leaky_relu,
        'tanh': tf.nn.tanh,
        'sigmoid': tf.nn.sigmoid,
        'softmax': tf.nn.softmax,
        'elu': tf.nn.elu,
        'softplus': tf.nn.softplus,
        "swish": swish
    }
    
    if name is None:
        return None
    
    if name not in act:
        print('activation `{}` not supported.  {}'.format(
            name, str(list(act.keys()))
        ))
        exit()
    return act[name]


def build_logits(input_net, name="dnn", output_dim=1, l2=0.0, reused=False, is_training=True):
    logits_scope_name = "%s/logits" % name
    with tf.variable_scope(logits_scope_name, values=(input_net,), reuse=reused) as dnn_logits_scope:
        
        if l2 > 1e-6:
            # add l2 regular
            l2regular = tf.contrib.layers.l2_regularizer(l2)
            dnn_logits = tf.contrib.layers.fully_connected(
                input_net,
                output_dim,
                activation_fn=None,
                variables_collections=[name],
                weights_initializer=initializers.variance_scaling_initializer(),
                scope=dnn_logits_scope,
                weights_regularizer=l2regular,
                biases_regularizer=l2regular,
                reuse=reused)
        else:
            dnn_logits = tf.contrib.layers.fully_connected(
                input_net,
                output_dim,
                activation_fn=None,
                variables_collections=[name],
                weights_initializer=initializers.variance_scaling_initializer(),
                scope=dnn_logits_scope,
                reuse=reused)
        tf_summary.add_hidden_layer_summary(dnn_logits, logits_scope_name)
        tf_summary.add_net_abs_mean_summary(dnn_logits, logits_scope_name)
    
    return dnn_logits


def scaled_dot_product_attention(q, k, v, mask=None):
    """
    使用 dk（向量维度）缩放的 self attention
    :param q: [batch, seq_len, embed_dim]
    :param k: [batch, seq_len, embed_dim]
    :param v: [batch, seq_len, embed_dim]
    :param mask: [batch, seq_len, embed_dim]
    :return: [batch, seq_len, embed_dim]
    """
    # query key 相乘获取匹配关系
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    # 使用dk进行缩放
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.sqrt(dk)
    # 掩码
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    # 权重归一化
    attention_weights = tf.nn.softmax(scaled_attention_logits, dim=-1)
    # attention 权重乘上value
    output = tf.matmul(attention_weights, v)
    return output, attention_weights
