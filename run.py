# coding=utf-8
"""
add following parameters for local run.
--env_str=local,critic,conf/param.json
"""
import tensorflow as tf
from modules.tf_flags import FLAGS
from util.conf_parser import ConfParser
from tensorflow.python.platform import gfile

envs = tuple(FLAGS.env_str.split(','))
cp = ConfParser(*envs)
print cp.to_string()
for k, v in ConfParser(*envs).parameters.items():
    setattr(FLAGS, k, v)


def main(_):
    if FLAGS.mode == "train" and FLAGS.continue_train == "false":
        print("start delete checkpoint_dir and summary_dir...")
        try:
            gfile.DeleteRecursively(FLAGS.path['checkpoint_dir'])
            gfile.DeleteRecursively(FLAGS.path['summary_dir'])
        except:
            print("delete checkpoint_dir and summary_dir error")
    
    __import__(name='modules.%s' % FLAGS.mode, fromlist=[FLAGS.mode]).run(FLAGS)


if __name__ == "__main__":
    tf.app.run()
