import tensorflow as tf
import collections


class MetricsRT(object):
    """ Real time metrics on tensorboard & console """
    
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.is_valid_task = bool(FLAGS.task_index == 0 and FLAGS.mode == 'train')
        
        self.log_ops = collections.OrderedDict()
        self.run_ops = collections.OrderedDict()
        
        with tf.name_scope('auc') as scope:
            self.scope = scope
    
    def add_scalar(self, op, name, to_log=True):
        if self.is_valid_task:
            with tf.name_scope(self.scope):
                tf.summary.scalar(name=name, tensor=op)
        
        if to_log:
            self.log_ops[name] = op
    
    def add_histogram(self, op, name):
        if self.is_valid_task:
            with tf.name_scope(self.scope):
                tf.summary.histogram(name=name, values=op)
    
    def add_auc(self, pred, label, mask=None, name='auc'):
        with tf.name_scope(self.scope):
            auc_op, update_op = tf.metrics.auc(
                labels=label,
                predictions=pred,
                num_thresholds=2000,
                weights=mask,
                name=name
            )
            if self.is_valid_task:
                tf.summary.scalar(name=name, tensor=auc_op)
        
        self.log_ops[name] = auc_op
        self.run_ops[name] = update_op
    
    def add_accuracy(self, pred, label, mask=None, name='accuracy'):
        with tf.name_scope(self.scope):
            with tf.name_scope(name):
                corrects = tf.equal(tf.argmax(pred, axis=-1), tf.cast(label, tf.int64))
                accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))
        
        self.log_ops[name] = accuracy
        return accuracy
    
    def merge_all(self):
        if self.is_valid_task:
            self.merge = tf.summary.merge_all()
