# coding=utf-8

import tensorflow as tf
import traceback
from transforms.tensor_op import pad_sparse2dense
import json


def format_feature_offline(features, config, item_num=10, seq_len=10):
    """
    item侧特征以 "|" 分割多个商品，user侧特征以 "|" 分割序列长度
    :param config: fg.json 配置
    :param item_num: 商品数量
    :param seq_len: 序列长度
    :return:
        item侧：shape [B, N]
        use侧：shape [B, S]
    """
    feature_dict = {}
    if not isinstance(config, dict):
        config = get_json_conf_from_file(config)
    feature_conf_dict = get_feature_conf_dict(config)
    for name, tensor in features.items():
        try:
            print("raw:### %s, tensor.dtype = %s, tensor.dense_shape= %s" % (name, tensor.dtype, tensor.get_shape()))
            if isinstance(tensor, tf.SparseTensor):
                if 'delimiter' in feature_conf_dict[name]:
                    delimiter = feature_conf_dict[name]['delimiter']
                else:
                    delimiter = "|"
                s_seq_len = feature_conf_dict[name]['seq_len'] if 'seq_len' in feature_conf_dict[name] else seq_len
                # print("delimiter:", delimiter)
                if feature_conf_dict[name]['expression'].startswith("user:"):
                    # offline use "|"
                    feature_value = pad_sparse2dense(tensor, default_value=delimiter.join(['0']*s_seq_len), pad_len=1)
                    feature_value = tf.reshape(feature_value, [-1])
                    feature_value = tf.string_split(feature_value, delimiter, skip_empty=False)
                    tensor = pad_sparse2dense(feature_value, default_value='0', pad_len=s_seq_len)
                else:
                    feature_value = pad_sparse2dense(tensor, default_value=delimiter.join(['0']*item_num), pad_len=1)
                    feature_value = tf.reshape(feature_value, [-1])
                    feature_value = tf.string_split(feature_value, delimiter, skip_empty=False)
                    tensor = pad_sparse2dense(feature_value, default_value='0', pad_len=item_num)
                if feature_conf_dict[name]['value_type'].lower() == "int":
                    tensor = tf.cast(tf.string_to_number(tensor), tf.int32)
                elif feature_conf_dict[name]['value_type'].lower() in ["float", "double"]:
                    tensor = tf.string_to_number(tensor)
            if name == 'search_pos':
                tensor = tensor / 10
            feature_dict[name] = tf.identity(tensor, name='%s_assign' % name)
        except:
            traceback.print_exc()
            print("error during format online feature: %s, tensor: %s" % (name, tensor))
    return feature_dict


def get_feature_conf_dict(conf):
    feature_conf_dict = {}
    for item in conf["features"]:
        feature_conf_dict[item['feature_name']] = item
    return feature_conf_dict


def get_json_conf_from_file(file_name):
    with open(file_name, 'r') as fp:
        config = json.load(fp)
    return config
