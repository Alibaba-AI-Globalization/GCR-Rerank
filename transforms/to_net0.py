import tensorflow as tf
from . import TransformAttr
import json


def to_net0_dict(feature_dict, conf, list_size=10):
    net0_dict = {}
    for feature_type in conf['features']:
        feature_list = conf['features'][feature_type]
        if len(feature_list) <= 0:
            continue
        tensor_list = []
        for name in feature_list:
            if name == 'search_pos':
                feature_dict[name] = tf.reshape(feature_dict[name], [-1, feature_dict[name].shape[-1]])
            if name.endswith('_same') or name.endswith('_bigger'):
                dim = tf.shape(feature_dict[name])[1]
                feature_dict[name] = tf.reshape(tf.pad(feature_dict[name], [[0, 0], [0, list_size - dim]]),
                                                [-1, list_size])
            tensor_list.append(feature_dict[name])
        net0_dict[feature_type] = tf.concat(tensor_list, axis=-1)
    
    return net0_dict
