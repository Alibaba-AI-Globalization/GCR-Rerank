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


def critic_predict(FLAGS):
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
        test_tables = [FLAGS.data[FLAGS.model + "_test"]]
        batch_size = conf['train']['batch_size']
        log_step = conf['train']['log_step']
        
        with tf.device('/job:worker/task:%d/cpu:0' % FLAGS.task_index):
            print('task index = %d' % FLAGS.task_index)
            dataset = OdpsData(columns='search_id,rn,features,label', defaults=['', '', '', ''])
            
            data = dataset.get_batch_data(test_tables, slice_id=FLAGS.task_index, slice_count=worker_count,
                                          num_epoch=1, batch_size=batch_size)
        
        with tf.device(tf.train.replica_device_setter(
                worker_device='/job:worker/task:%d' % FLAGS.task_index,
                cluster=cluster)):
            
            global_step = tf.Variable(0, name='global_step', trainable=False)
            model = importlib.import_module('models.%s' % FLAGS.model_conf['name'].lower()).Model(FLAGS)
            write_op, close_writer_op = model.get_outputs(data)
        
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(FLAGS.task_index == 0), hooks=None,
                                               checkpoint_dir=FLAGS.path['checkpoint_dir'],
                                               save_checkpoint_secs=FLAGS.distribution['checkpoint_sec'],
                                               ) as mon_sess:
            print('checkpoint dir: {}'.format(FLAGS.path['checkpoint_dir']))
            print('-' * 90 + '\n\tMonitoredTrainingSession starting... @ %s\n' % datetime.datetime.now() + "-" * 90)
            print('globalStep: %d' % mon_sess.run(global_step))
            
            try:
                step = 0
                while not mon_sess.should_stop():
                    step += 1
                    mon_sess.run(write_op)
                    if step == 1 or step % log_step == 0:
                        print('test step: %d, global_step: %d' % (step, mon_sess.run(global_step)))
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
                mon_sess.run(close_writer_op)


def generator_predict(FLAGS):
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
        test_tables = [FLAGS.data[FLAGS.model + "_test"]]
        batch_size = conf['train']['batch_size']
        log_step = conf['train']['log_step']
        
        with tf.device('/job:worker/task:%d/cpu:0' % FLAGS.task_index):
            print('task index = %d' % FLAGS.task_index)
            dataset = OdpsData(columns='search_id,rn,features,label', defaults=['', '', '', ''])
            
            data = dataset.get_batch_data(test_tables, slice_id=FLAGS.task_index, slice_count=worker_count,
                                          num_epoch=1, batch_size=batch_size)
        
        with tf.device(tf.train.replica_device_setter(
                worker_device='/job:worker/task:%d' % FLAGS.task_index,
                cluster=cluster)):
            
            global_step = tf.Variable(0, name='metrics_predict_global', trainable=False)
            global_step_add = tf.assign_add(global_step, 1, use_locking=True)
            model = importlib.import_module('models.%s' % FLAGS.model_conf['name'].lower()).Model(FLAGS)
            metrics_tensor = model.get_outputs(data)

            variables = tf.all_variables()
            model_variables = []
            new_variables = []
            for v in variables:
                if not v.name.startswith('metrics'):
                    print 'restore var: ', v.name
                    model_variables.append(v)
                else:
                    new_variables.append(v)
            saver = tf.train.Saver(model_variables)
            new_init_op = tf.variables_initializer(new_variables)
        
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(FLAGS.task_index == 0)
                                               ) as mon_sess:
            print('-' * 90 + '\n\tMonitoredTrainingSession starting... @ %s\n' % datetime.datetime.now() + "-" * 90)
            if FLAGS.task_index == 0:
                mon_sess.run(new_init_op)
            # load model
            print('checkpoint_dir: {}'.format(FLAGS.path['checkpoint_dir']))
            saver.restore(mon_sess, tf.train.latest_checkpoint(FLAGS.path['checkpoint_dir']))
            checkpoint_critic = FLAGS.path['checkpoint_critic']
            if tf.gfile.IsDirectory(checkpoint_critic):
                checkpoint_critic = tf.train.latest_checkpoint(checkpoint_critic)
            print("restore critic model from path: {}".format(checkpoint_critic))
            model.critic_saver.restore(mon_sess, checkpoint_critic)
            
            try:
                step = 0
                while not mon_sess.should_stop():
                    step += 1
                    metrics_tensor['global_step'] = global_step_add
                    metrics = mon_sess.run(metrics_tensor)
                    if step % log_step == 0:
                        print('test step: %d, global_step: %d' % (step, mon_sess.run(global_step)))
                        
                        print("")
                        gs = metrics['global_step']
                        print('step: %d, global_step: %d' % (step, gs))
                        # generator base state
                        print("overview:")
                        print 'avg slate replace ratio: \t', (metrics['higherSum'] / (batch_size * gs))
                        print 'avg item replace ratio: \t', (metrics['replaceItemSum'] / (batch_size * 10 * gs))
                        print 'avg gen slate asp gap: \t', (metrics['genPriceSum'] / metrics['originPriceSum']) - 1
                        print 'avg cr gap: \t', (metrics['diffScoreSum'] / metrics['originScoreSum'])
                        print 'avg diff score: \t', (metrics['diffScoreSum'] / gs)
                        print 'avg origin score: \t', (metrics['originScoreSum'] / gs)
                        print 'keep pay ratio: \t', (metrics['genPaySum'] / metrics['oriPaySum'])
                        print 'origin pay num: \t', metrics['oriPaySum']
                        # # diversity
                        # print("diversity:")
                        # print 'avg origin diversity: \t', (metrics['ori_diversity_sum'] / (batch_size * gs))
                        # print 'avg gen diversity: \t', (metrics['gen_diversity_sum'] / (batch_size * gs))
                        # print 'avg origin diversity count: \t', (metrics['ori_diver_dist_sum'])
                        # print 'avg gen    diversity count: \t', (metrics['gen_diver_dist_sum'])
                        # print 'avg origin diversity dist: \t', (metrics['ori_diver_dist_ratio'])
                        # print 'avg gen    diversity dist: \t', (metrics['gen_diver_dist_ratio'])
                        # # relevance
                        # print("relevance:")
                        # print 'ori CateLevelScore ratio: \t', (metrics['ori_cate_score_sum'] / (batch_size * gs))
                        # print 'gen CateLevelScore ratio: \t', (metrics['gen_cate_score_sum'] / (batch_size * gs))
                        # print 'ori RecallByOri ratio: \t', (metrics['ori_recall_ori_sum'] / (batch_size * gs))
                        # print 'gen RecallByOri ratio: \t', (metrics['gen_recall_ori_sum'] / (batch_size * gs))
                        # price>0 & cr>0
                        print("price>0 & cr>0:")
                        print 'avg replace ratio: \t', (metrics['crHigherNum'] / (batch_size * gs))
                        print 'avg cr gap: \t', (metrics['cr_cr_gap_sum'] / (batch_size * gs))
                        print 'avg asp gap: \t', (metrics['crPriceSum'] / metrics['originPriceSum']) - 1
                        print 'cr replace cr gap: \t', (metrics['cr_cr_gap_sum'] / metrics['crHigherNum'])
                        print 'cr replace asp gap: \t', (metrics['cr_asp_gap_sum'] / metrics['crHigherNum'])
                        # price>0 & gmv_gap>0
                        print("price>0 & gmv_gap>0:")
                        print 'avg replace ratio: \t', (metrics['gmvHigherSum'] / (batch_size * gs))
                        print 'avg cr gap: \t', (metrics['gmv_cr_gap_sum'] / (batch_size * gs))
                        print 'avg asp gap: \t', (metrics['gmvPriceSum'] / metrics['originPriceSum']) - 1
                        print 'gmv replace cr gap: \t', (metrics['gmv_cr_gap_sum'] / metrics['gmvHigherSum'])
                        print 'gmv replace asp gap: \t', (metrics['gmv_asp_gap_sum'] / metrics['gmvHigherSum'])
                        # price>0
                        print("price>0:")
                        print 'avg replace ratio: \t', (metrics['priceHigherNum'] / (batch_size * gs))
                        print 'avg cr gap: \t', (metrics['price_cr_gap_sum'] / (batch_size * gs))
                        print 'avg asp gap: \t', (metrics['pricePriceSum'] / metrics['originPriceSum']) - 1
                        print 'price replace cr gap: \t', (metrics['price_cr_gap_sum'] / metrics['priceHigherNum'])
                        print 'price replace asp gap: \t', (metrics['price_asp_gap_sum'] / metrics['priceHigherNum'])

                    if step > 2000:
                        break
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


def generator_predict_to_odps(FLAGS):
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
        test_tables = [FLAGS.data[FLAGS.model + "_test"]]
        batch_size = conf['train']['batch_size']
        log_step = conf['train']['log_step']
        
        with tf.device('/job:worker/task:%d/cpu:0' % FLAGS.task_index):
            print('task index = %d' % FLAGS.task_index)
            dataset = OdpsData(columns='search_id,rn,features,label', defaults=['', '', '', ''])
            
            data = dataset.get_batch_data(test_tables, slice_id=FLAGS.task_index, slice_count=worker_count,
                                          num_epoch=1, batch_size=batch_size)
        
        with tf.device(tf.train.replica_device_setter(
                worker_device='/job:worker/task:%d' % FLAGS.task_index,
                cluster=cluster)):
            
            global_step = tf.Variable(0, name='metrics_predict_global', trainable=False)
            global_step_add = tf.assign_add(global_step, 1, use_locking=True)
            model = importlib.import_module('models.%s' % FLAGS.model_conf['name'].lower()).Model(FLAGS)
            metrics_tensor = model.get_outputs(data)
            write_op, close_writer_op = model.to_odps(data)
            
            variables = tf.all_variables()
            model_variables = []
            new_variables = []
            for v in variables:
                if not v.name.startswith('metrics'):
                    print 'restore var: ', v.name
                    model_variables.append(v)
                else:
                    new_variables.append(v)
            saver = tf.train.Saver(model_variables)
            new_init_op = tf.variables_initializer(new_variables)
        
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(FLAGS.task_index == 0)
                                               ) as mon_sess:
            print('-' * 90 + '\n\tMonitoredTrainingSession starting... @ %s\n' % datetime.datetime.now() + "-" * 90)
            if FLAGS.task_index == 0:
                mon_sess.run(new_init_op)
            # load model
            print('checkpoint_dir: {}'.format(FLAGS.path['checkpoint_dir']))
            saver.restore(mon_sess, tf.train.latest_checkpoint(FLAGS.path['checkpoint_dir']))
            checkpoint_critic = FLAGS.path['checkpoint_critic']
            if tf.gfile.IsDirectory(checkpoint_critic):
                checkpoint_critic = tf.train.latest_checkpoint(checkpoint_critic)
            print("restore critic model from path: {}".format(checkpoint_critic))
            model.critic_saver.restore(mon_sess, checkpoint_critic)
            
            try:
                step = 0
                while not mon_sess.should_stop():
                    step += 1
                    mon_sess.run(write_op)
                    if step % 100 == 0:
                        print("step=%d" % step)
                    # if step > 1000:
                    #     break

            except tf.errors.OutOfRangeError:
                print("predict end")
            except:
                traceback.print_exc()
            finally:
                print("close WriterOp")
                mon_sess.run(close_writer_op)
            return


def generator_predict_ips(FLAGS):
    print('start generator_predict IPS ... ')
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
        test_tables = [FLAGS.data[FLAGS.model + "_test"]]
        batch_size = conf['train']['batch_size']
        log_step = conf['train']['log_step']
    
        with tf.device('/job:worker/task:%d/cpu:0' % FLAGS.task_index):
            print('task index = %d' % FLAGS.task_index)
            dataset = OdpsData(columns='search_id,rn,features,label', defaults=['', '', '', ''])
        
            data = dataset.get_batch_data(test_tables, slice_id=FLAGS.task_index, slice_count=worker_count,
                                          num_epoch=1, batch_size=batch_size)
    
        with tf.device(tf.train.replica_device_setter(
                worker_device='/job:worker/task:%d' % FLAGS.task_index,
                cluster=cluster)):
        
            global_step = tf.Variable(0, name='metrics_predict_global', trainable=False)
            global_step_add = tf.assign_add(global_step, 1, use_locking=True)
            model = importlib.import_module('models.%s' % FLAGS.model_conf['name'].lower()).Model(FLAGS)

            candidate_size = 50
            # feature preprocess
            with tf.name_scope("featureparser_feature_input"):
                fg_configs = get_json_conf_from_file(FLAGS.feature_conf['input'])
                featureparser_features = featureparser_fg.parse_genreated_fg(fg_configs, data['features'])
                featureparser_features = format_feature_offline(featureparser_features, FLAGS.feature_conf['input'],
                                                      item_num=candidate_size)
                # label
                label_configs = get_json_conf_from_file(FLAGS.feature_conf['label'])
                label_dict = featureparser_fg.parse_genreated_fg(label_configs, data['label'])
                label_dict = format_feature_offline(label_dict, FLAGS.feature_conf['label'])
                # slate score
                other_features = featureparser_fg.parse_genreated_fg(label_configs, data['features'])
                other_features = format_feature_offline(other_features, FLAGS.feature_conf['label'],
                                                        item_num=candidate_size)

            # only offline item_num is candidate_size
            item_num = candidate_size
            # build model
            net0_dict = model.get_net0_dict(featureparser_features)
            batch_size = tf.shape(data['rn'])[0]
            model.pre_index = tf.tile(tf.reshape(tf.cast(tf.range(10), dtype=tf.int64), [1, 10]), [batch_size, 1])
            model.label_dict = label_dict
            model.build_network(net0_dict)
            metrics = model.calc_metrics(model.get_var_dict(), featureparser_features)
            model.load_critic_variables()

            # gen_score = tf.reduce_join(tf.as_string(model.finalScoreState), axis=1, separator="|")  # [B,1]
            # seleted_index = tf.reduce_join(tf.as_string(model.finalLoopState), axis=1, separator="|")
            seleted_index = tf.as_string(model.finalLoopState)
            gen_score = model.finalScoreState
            click_label = label_dict['click'][:, :10]
            pay_label = label_dict['pay'][:, :10]
            gmv = other_features['gmv'][:, :10]
            # slate_score = other_features['featureparser_trace_slate_score'][:, :10]
            # compare two critic scores, origin(score), generator(score)
            predict_result_table = FLAGS.data['critic_result_table']
            sid = tf.tile(tf.reshape(data['search_id'], [-1, 1]), [1, 10])
            rn = tf.tile(tf.reshape(data['rn'], [-1, 1]), [1, 10])
            writer = tf.TableRecordWriter(predict_result_table, slice_id=FLAGS.task_index)
            writeOp = writer.write([0, 1, 2, 3, 4, 5, 6, 7],
                                   [sid, rn, seleted_index, gmv, gen_score, pay_label, gen_score, click_label])
            closeWriterOp = writer.close()

        variables = tf.all_variables()
        model_variables = []
        new_variables = []
        for v in variables:
            if not v.name.startswith('metrics'):
                print 'restore var: ', v.name
                model_variables.append(v)
            else:
                new_variables.append(v)
        saver = tf.train.Saver(model_variables)
    
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(FLAGS.task_index == 0)
                                               ) as mon_sess:
            print('-' * 90 + '\n\tMonitoredTrainingSession starting... @ %s\n' % datetime.datetime.now() + "-" * 90)
            # load model
            print('checkpoint_dir: {}'.format(FLAGS.path['checkpoint_dir']))
            saver.restore(mon_sess, tf.train.latest_checkpoint(FLAGS.path['checkpoint_dir']))
            checkpoint_critic = FLAGS.path['checkpoint_critic']
            if tf.gfile.IsDirectory(checkpoint_critic):
                checkpoint_critic = tf.train.latest_checkpoint(checkpoint_critic)
            print("restore critic model from path: {}".format(checkpoint_critic))
            model.critic_saver.restore(mon_sess, checkpoint_critic)
            print("predict result save in: {}".format(predict_result_table))

            try:
                step = 0
                while not mon_sess.should_stop():
                    step += 1
                    mon_sess.run(writeOp)
                    if step % 100 == 0:
                        print("step=%d" % step)
                    # if step > 200:
                    #     break
            except tf.errors.OutOfRangeError:
                print("predict end")
            except:
                traceback.print_exc()
            finally:
                print("close WriterOp")
                mon_sess.run(closeWriterOp)
            return


def get_model_args(FLAGS):
    if FLAGS.model == "critic":
        conf = FLAGS.critic_model
    elif FLAGS.model == "generator":
        conf = FLAGS.generator_model
    setattr(FLAGS, 'model_conf', conf)
    return conf


def run(FLAGS):
    if FLAGS.model == "critic":
        critic_predict(FLAGS)
    elif FLAGS.model == "generator":
        if hasattr(FLAGS, "g_result") and FLAGS.g_result is True:
            generator_predict_to_odps(FLAGS)
        elif FLAGS.generator_model['train']['isp_mode'] is True:
            generator_predict_ips(FLAGS)
        else:
            generator_predict(FLAGS)
