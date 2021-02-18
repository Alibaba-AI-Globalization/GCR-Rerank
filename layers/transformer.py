# coding=utf-8
import tensorflow as tf
from layer_lib import scaled_dot_product_attention, build_dnn_net


class MutilHeadAttention(object):
    """
	multi head attention
    """
    # 构造 mutil head attention层
    
    def __init__(self, dim, num_heads):
        """
        :param dim: 输出的维度
        :param num_heads: head 数量
        """
        self.num_heads = num_heads
        self.dim = dim
        
        # dim 必须可以正确分为各个头
        assert dim % num_heads == 0
        # 分头后的维度
        self.depth = dim // num_heads
    
    def split_heads(self, x, batch_size):
        # 分头, 将头个数的维度 放到 seq_len 前面
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def __call__(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]
        
        # 分头前的前向网络，获取q、k、v语义
        # (batch_size, seq_len, dim)
        q = build_dnn_net(q, name="linear_q", hidden_units=[self.dim], activation=[None], reused=tf.AUTO_REUSE)
        k = build_dnn_net(q, name="linear_k", hidden_units=[self.dim], activation=[None], reused=tf.AUTO_REUSE)
        v = build_dnn_net(q, name="linear_v", hidden_units=[self.dim], activation=[None], reused=tf.AUTO_REUSE)
        
        # 分头
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # 通过缩放点积注意力层
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        # 把多头维度后移
        scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3])  # (batch_size, seq_len_v, num_heads, depth)
        
        # 合并多头
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.dim))
        
        # 全连接重塑
        output = build_dnn_net(concat_attention, name="linear_output", hidden_units=[self.dim], activation=[None],
                               reused=tf.AUTO_REUSE)
        return output, attention_weights
