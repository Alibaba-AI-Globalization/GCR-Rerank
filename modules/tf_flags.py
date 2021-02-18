# -*- coding: utf-8 -*-
import tensorflow as tf


flags = tf.app.flags

flags.DEFINE_string('env_str', '', '')
flags.DEFINE_string('tag', '', '')
flags.DEFINE_string('continue_train', 'false', 'continue train')

flags.DEFINE_string('ps_hosts', '', 'ps hosts')
flags.DEFINE_string('worker_hosts', '', 'worker hosts')
flags.DEFINE_string('job_name', None, 'job name: worker or ps')
flags.DEFINE_integer('task_index', None, 'Worker task index')

FLAGS = tf.app.flags.FLAGS
