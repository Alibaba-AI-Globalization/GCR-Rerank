# coding=utf-8
"""
N：表示一个Slate的大小，即一个Slate的商品数量
B：表示batch size
S：表示sequence length
E：表示embedding 维度大小
"""
import tensorflow as tf
import json
from transforms import tensor_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope as vs
from . import TransformAttr


class Preprocess(object):
    
    def __init__(self, conf_file):
        print("Preprocess use config file: {}".format(conf_file))
        self.feature_conf_dict = None
        self.copy_conf_dict = None
        self.need_extend_features = None
        self.parse_config(conf_file)

        self.batch_size = None
        self.feature_dict = {}
        self.embedding_dict = {}
    
    def parse_config(self, conf_file):
        with open(conf_file, "r") as fp:
            conf = json.load(fp)
        
        self.need_extend_features = conf['need_extend_features']
        
        feature_conf_dict = {}
        copy_conf_dict = {}
        for item in conf["preprocess"]:
            if item['transform_name'] == "copy":
                copy_conf_dict[item['feature_name']] = item
            else:
                feature_conf_dict[item['feature_name']] = item
        self.feature_conf_dict = feature_conf_dict
        self.copy_conf_dict = copy_conf_dict
    
    def process(self, feature_dict, list_size=10, variable_scope="critic_network"):
        """
        input:
            item: [B, N]
            user: [B, 1] or [B, S]
        output:
            item: [B*N, 1]
            user: [B, 1, E] or [B, S, E]
        """
        # get batch size
        for feature_name, tensor in feature_dict.items():
            self.batch_size = tf.shape(tensor)[0]
            break
        
        # copy tensor
        for feature_name, feature_value in feature_dict.items():
            if feature_name in self.copy_conf_dict:
                output_name = self.copy_conf_dict[feature_name].get(TransformAttr.output_name)
                print "copy feature_name:", feature_name, output_name
                feature_dict[output_name] = feature_value
        
        # normalization
        normal_feature_dict = {}
        for feature_name, feature_value in feature_dict.items():
            normal_feature_dict[feature_name] = self.float_tensor_normalization(feature_name, feature_value)
            
            if (feature_name in self.feature_conf_dict
                    and self.feature_conf_dict[feature_name]['expression'].startswith("user:")):
                self.feature_dict[feature_name] = normal_feature_dict[feature_name]
            else:
                self.feature_dict[feature_name] = tf.reshape(normal_feature_dict[feature_name], [-1, 1])
        
        # extend context features
        for feature_name in self.need_extend_features:
            if feature_name in normal_feature_dict:
                self.extend_feature(feature_name, normal_feature_dict[feature_name], list_size=list_size)
            else:
                print("feature: {} is not exist!".format(feature_name))
        
        # embedding
        embed_feature_dict = {}
        for name in self.feature_dict:
            if name not in self.feature_conf_dict:
                continue
            transform_name = self.feature_conf_dict[name].get(TransformAttr.transform_name)
            if transform_name in ("id_embedding", "share_id_embedding"):
                self.add_id_embedding(self.feature_conf_dict[name], variable_scope)
                embed = self.embedding_lookup(self.feature_dict[name], self.feature_conf_dict[name])
                if self.feature_conf_dict[name]['expression'].startswith("user:"):
                    embed_feature_dict[name] = embed
                else:
                    embed_feature_dict[name] = tf.reshape(embed, [-1, embed.shape[-1]])
            elif transform_name == 'id_one_hot':
                bucket_size = self.feature_conf_dict[name].get(TransformAttr.bucket_size)
                id_feature = tf.reshape(self.feature_dict[name], [-1, 1])
                dim0 = tf.shape(id_feature)[0]
                indices = tf.concat([tf.expand_dims(tf.range(0, dim0), 1), id_feature], axis=1)
                id_one_hot = tf.sparse_to_dense(indices, tf.stack([dim0, bucket_size]), 1.0, 0.0)
                self.feature_dict[name] = id_one_hot
            elif transform_name == "hash_id_embedding":
                self.add_id_embedding(self.feature_conf_dict[name], variable_scope)
                embed = self.hash_embedding_lookup(self.feature_dict[name], self.feature_conf_dict[name])
                if self.feature_conf_dict[name]['expression'].startswith("user:"):
                    embed_feature_dict[name] = embed
                else:
                    embed_feature_dict[name] = tf.reshape(embed, [-1, embed.shape[-1]])
        for name in embed_feature_dict:
            self.feature_dict[name] = embed_feature_dict[name]
        
        return self.feature_dict
    
    def extend_feature(self, feature_name, feature_value, list_size=10):
        if feature_value.dtype == tf.int32 or feature_value.dtype == tf.int64:
            tensor_same = tensor_op.compare_same(feature_value, n=list_size)
            tensor_same = tf.reshape(tensor_same, [-1, list_size])
            
            self.feature_dict[feature_name + "_same"] = tf.cast(tensor_same, dtype=tf.float32)
        else:
            tensor_mean, tensor_std = tensor_op.float_context_statistic(feature_value, item_num=list_size)
            tensor_bias = tensor_op.calc_bias(feature_value, tensor_mean, tensor_std)
            tensor_bigger = tensor_op.calc_bigger(feature_value)
            tensor_order = tensor_op.argsort(tensor_bigger)
            tensor_mean = tf.clip_by_value(tensor_mean, -5., 5.)
            tensor_std = tf.clip_by_value(tensor_std, -5., 5.)
            tensor_bias = tf.clip_by_value(tensor_bias, -5., 5.)
            tensor_order = tf.clip_by_value(tensor_order, -5., 5.)
            
            self.feature_dict[feature_name + "_mean"] = tf.reshape(tensor_mean, [-1, 1])
            self.feature_dict[feature_name + "_std"] = tf.reshape(tensor_std, [-1, 1])
            self.feature_dict[feature_name + "_bias"] = tf.reshape(tensor_bias, [-1, 1])
            self.feature_dict[feature_name + "_bigger"] = tf.reshape(tensor_bigger, [-1, list_size])
            self.feature_dict[feature_name + "_order"] = tf.reshape(tensor_order, [-1, 1])
    
    def float_tensor_normalization(self, name, tensor):
        if name not in self.feature_conf_dict:
            print("feature: [{}] no preprocess config".format(name))
            return tensor
        fc = self.feature_conf_dict[name]
        scale_mean = fc.get("scale_mean")
        scale_stddev = fc.get("scale_stddev")
        scale_median = fc.get("scale_median")
        scale_coef = fc.get("scale_coef")
        # 1. normalization as stddev and mean
        if scale_mean is not None and scale_stddev is not None:
            tensor = tf.to_float(tensor)
            tensor = (tensor - scale_mean) / scale_stddev
        # 2. normalization as log( (1+v)/(1+median) )
        elif scale_mean is not None:
            tensor = tf.log((1 + tensor) / (1 + scale_median))
        elif scale_coef is not None:
            tensor = scale_coef * tensor
        
        max_clip = fc.get("clip_max")
        min_clip = fc.get("clip_min")
        if max_clip is not None and min_clip is not None:
            tensor = tf.clip_by_value(tensor, min_clip, max_clip)
        return tensor
    
    def add_id_embedding(self, fc, embedding_scope):
        embedding_name = fc.get(TransformAttr.embedding_name)
        if embedding_name is None:
            embedding_name = fc.get(TransformAttr.feature_name) + '_embedding'
        bucket_size = fc.get(TransformAttr.bucket_size)
        embedding_size = fc.get(TransformAttr.dimension)
        
        initializer = fc.get(TransformAttr.initializer)
        with tf.variable_scope(embedding_scope, reuse=tf.AUTO_REUSE):
            if initializer and initializer.startswith("uniform"):
                val = float(initializer.split(":")[1])
                rand_initializer = tf.random_uniform_initializer(-val, val)
            elif initializer and initializer.startswith("variance_scaling"):
                rand_initializer = tf.variance_scaling_initializer()
            else:
                rand_initializer = tf.random_uniform_initializer(-1.0, 1.0)
            embeddings = vs.get_variable(embedding_name, [bucket_size, embedding_size],
                                         initializer=rand_initializer,
                                         collections=[ops.GraphKeys.GLOBAL_VARIABLES,
                                                      ops.GraphKeys.MODEL_VARIABLES],
                                         trainable=fc.get(TransformAttr.trainable))
        
        self.embedding_dict[embedding_name] = embeddings
        return embeddings
    
    def embedding_lookup(self, id_tensor, fc):
        embedding_name = fc.get(TransformAttr.embedding_name)
        if embedding_name is None:
            embedding_name = fc.get(TransformAttr.feature_name) + '_embedding'
        bucket_size = fc.get(TransformAttr.bucket_size)
        embedding_size = fc.get(TransformAttr.dimension)
        
        pad = tf.fill(tf.shape(id_tensor), value=0)
        id_tensor = tf.where(tf.logical_or(id_tensor >= bucket_size, id_tensor < 0), pad, id_tensor)
        embeddings = self.embedding_dict[embedding_name]
        embed = tf.nn.embedding_lookup(embeddings, id_tensor)
        
        return embed

    def hash_embedding_lookup(self, tensor, fc):
        bucket_size = fc.get(TransformAttr.bucket_size)
        embedding_name = fc.get(TransformAttr.embedding_name)
        if embedding_name is None:
            embedding_name = fc.get(TransformAttr.feature_name) + '_embedding'
        
        id_tensor = tf.string_to_hash_bucket_fast(tensor, bucket_size)
        embeddings = self.embedding_dict[embedding_name]
        embed = tf.nn.embedding_lookup(embeddings, id_tensor)
        
        return embed
