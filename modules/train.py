import tensorflow as tf
from dataset.data import OdpsData
import json
import importlib
from transforms.data_format import format_feature_offline
from transforms.data_format import get_json_conf_from_file
from transforms.preprocessing import Preprocess
import datetime
import traceback
import featureparser_fg


def cluster_train(FLAGS):
    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    cluster = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    worker_count = len(worker_hosts)
    print('worker count: {}, job name: {}, task index: {}'.format(worker_count, FLAGS.job_name, FLAGS.task_index))
    
    if FLAGS.job_name == 'ps':
        server.join()
    elif FLAGS.job_name == 'worker':
        conf = get_model_args(FLAGS)
        train_tables = [FLAGS.data[FLAGS.model + "_train"]]
        valid_tables = [FLAGS.data[FLAGS.model + "_valid"]]
        epoch = conf['train']['epoch']
        batch_size = conf['train']['batch_size']
        train_step = conf['train']['train_step']
        summary_step = conf['train']['summary_step']
        log_step = conf['train']['log_step']
        
        with tf.device('/job:worker/task:%d/cpu:0' % FLAGS.task_index):
            print('task index = %d' % FLAGS.task_index)
            dataset = OdpsData(columns='search_id,rn,features,label', defaults=['', '', '', ''])
            
            if FLAGS.task_index == 0:  # validate worker
                data = dataset.get_batch_data(valid_tables, slice_id=FLAGS.task_index, slice_count=1,
                                              num_epoch=None, batch_size=batch_size)
            else:  # training worker
                data = dataset.get_batch_data(train_tables, slice_id=FLAGS.task_index - 1, slice_count=worker_count - 1,
                                              num_epoch=epoch, batch_size=batch_size)
        
        with tf.device(tf.train.replica_device_setter(
                    worker_device='/job:worker/task:%d' % FLAGS.task_index,
                    cluster=cluster)):

            # build model
            global_step = tf.Variable(0, name='global_step', trainable=False)
            model = importlib.import_module('models.%s' % FLAGS.model_conf['name'].lower()).Model(FLAGS)
            model.build(data, global_step)
            
            # auc metric op
            mrt = model.mrt
            log_ops_names = list(mrt.log_ops.keys())
            log_ops = list(mrt.log_ops.values())
            run_ops = list(mrt.run_ops.values())
            
            if FLAGS.task_index == 0:
                summary_op = tf.summary.merge_all()
        
        hooks = [tf.train.StopAtStepHook(last_step=train_step)]
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(FLAGS.task_index == 0), hooks=hooks,
                                               checkpoint_dir=FLAGS.path['checkpoint_dir'],
                                               save_checkpoint_secs=FLAGS.distribution['checkpoint_sec'],
                                               ) as mon_sess:
            print('checkpoint dir: {}'.format(FLAGS.path['checkpoint_dir']))
            print('-' * 90 + '\n\tMonitoredTrainingSession starting... @ %s\n' % datetime.datetime.now() + "-" * 90)
            log_writer = tf.summary.MetricsWriter("./")
            if FLAGS.task_index == 0:
                summary_writer = tf.summary.FileWriter(FLAGS.path['summary_dir'], mon_sess.graph)
            if FLAGS.model == 'generator':
                checkpoint_critic = FLAGS.path['checkpoint_critic']
                if tf.gfile.IsDirectory(checkpoint_critic):
                    checkpoint_critic = tf.train.latest_checkpoint(checkpoint_critic)
                print("restore critic model from path: {}".format(checkpoint_critic))
                if model.critic_saver is not None:
                    model.critic_saver.restore(mon_sess, checkpoint_critic)
                if 'checkpoint_actor' in FLAGS.path:
                    checkpoint_actor = FLAGS.path['checkpoint_actor']
                    if tf.gfile.IsDirectory(checkpoint_actor):
                        checkpoint_actor = tf.train.latest_checkpoint(checkpoint_actor)
                    print("restore actor model from path: {}".format(checkpoint_actor))
                    if model.actor_saver is not None:
                        model.actor_saver.restore(mon_sess, checkpoint_actor)
            
            try:
                step = last_1000_loss = reward = 0
                start_time = datetime.datetime.now()
                while not mon_sess.should_stop():
                    step += 1
                    
                    if FLAGS.task_index == 0:
                        if FLAGS.model == 'generator':
                            res = mon_sess.run([global_step, summary_op] + log_ops + run_ops + [model.critic_slate_score])
                            reward += res[-1]
                        else:
                            res = mon_sess.run([global_step, summary_op] + log_ops + run_ops)
                        if step == 1 or step % summary_step == 0:
                            model_global_step = res[0]
                            summary = res[1]
                            print("step: %d, global_step: %d" % (step, model_global_step)),
                            for i, (name, value) in enumerate(zip(log_ops_names, res[2:2+len(log_ops_names)])):
                                print('%s: %.3f' % (name, value)),
                            print('')
                            summary_writer.add_summary(summary, model_global_step)
                            log_writer.add_scalar("reward", reward / summary_step, step)
                            reward = 0
                    else:
                        res = mon_sess.run(
                            [model.train_op, global_step, model.loss, model.loss_ema] + log_ops + run_ops)
                        
                        model_global_step = res[1]
                        last_1000_loss += res[2]
                        if step == 1 or step % log_step == 0:
                            last_1000_loss = last_1000_loss * log_step if step == 1 else last_1000_loss
                            end_time = datetime.datetime.now()
                            used_time = (end_time - start_time).total_seconds()
                            eta_hour = used_time / step * (train_step / (worker_count-1) - step) / 3600
                            if used_time < 3600:
                                print("step: {}, global_step: {}, avg_loss: {}, Cost: {:.2f}s, Eta: {:.2f}h".format(
                                    step, model_global_step, last_1000_loss / log_step, used_time, eta_hour))
                                for i, (name, value) in enumerate(zip(log_ops_names, res[4:4 + len(log_ops_names)])):
                                    print('%s: %.3f' % (name, value)),
                                print('')
                            else:
                                print("step: {}, global_step: {}, avg_loss: {}, Cost: {:.2f}h, Eta: {:.2f}h".format(
                                    step, model_global_step, last_1000_loss / log_step, used_time / 3600, eta_hour))
                                for i, (name, value) in enumerate(zip(log_ops_names, res[4:4 + len(log_ops_names)])):
                                    print('%s: %.3f' % (name, value)),
                                print('')
                            log_writer.add_scalar("avg_loss", last_1000_loss / log_step, step)
                            last_1000_loss = 0
            except tf.errors.OutOfRangeError:
                print('Data reading done!')
                if FLAGS.task_index == 0:
                    import time
                    eval_sess = mon_sess
                    while not isinstance(eval_sess, tf.Session):
                        eval_sess = eval_sess._sess
                    if worker_count > 1:
                        _g = eval_sess.run(global_step)
                        _ng = 0
                        while _g > _ng:
                            print('waiting for other workers...')
                            time.sleep(30)
                            _ng = _g
                            _g = eval_sess.run(global_step)
            finally:
                log_writer.close()
                if not mon_sess.should_stop():
                    mon_sess.request_stop(notify_all=True)
                if FLAGS.task_index == 0:
                    summary_writer.close()


def get_model_args(FLAGS):
    if FLAGS.model == "critic":
        conf = FLAGS.critic_model
    elif FLAGS.model == "generator":
        conf = FLAGS.generator_model
    setattr(FLAGS, 'model_conf', conf)
    return conf


def run(FLAGS):
    cluster_train(FLAGS)
