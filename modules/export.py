import tensorflow as tf
import json
import importlib
from transforms.data_format import get_json_conf_from_file
from tensorflow.python.platform import gfile
import datetime
import featureparser_fg
import os


def export(FLAGS):
    tf.get_default_graph().set_shape_optimize(False)
    features_str = tf.placeholder(dtype=tf.string, name='rerank_input')
    # feature preprocess
    fg_configs = get_json_conf_from_file(FLAGS.feature_conf['input'])
    
    print("online, very important, all float features need to be raw_feature avoid id_feature float -> int")
    for idx, val in enumerate(fg_configs["features"]):
        if val['value_type'].lower() != "string" and not val['expression'].startswith("user:"):
            fg_configs["features"][idx]["feature_type"] = "raw_feature"  # very important!!!
        else:
            print('id_features: %s' % val['feature_name'])
    
    featureparser_features = featureparser_fg.parse_genreated_fg(fg_configs, features_str)
    
    # build model
    global_step = tf.Variable(1000, name='global_step', trainable=False)
    model = importlib.import_module('models.%s' % FLAGS.online_model).Model(FLAGS)
    model.build_network(featureparser_features)
    
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        
        # load model
        checkpoint_critic = FLAGS.path['checkpoint_critic']
        checkpoint_generator = FLAGS.path['checkpoint_generator']
        checkpoint_actor = ''
        checkpoint_asp = FLAGS.path['checkpoint_asp_generator'] if 'checkpoint_asp_generator' in FLAGS.path else ''
        checkpoint_diversity = FLAGS.path['checkpoint_diversity'] if 'checkpoint_diversity' in FLAGS.path else ''
        if gfile.IsDirectory(FLAGS.path['checkpoint_critic']):
            checkpoint_critic = tf.train.latest_checkpoint(FLAGS.path['checkpoint_critic'])
        if gfile.IsDirectory(FLAGS.path['checkpoint_generator']):
            checkpoint_generator = tf.train.latest_checkpoint(FLAGS.path['checkpoint_generator'])
        if checkpoint_diversity != '' and gfile.IsDirectory(checkpoint_diversity):
            checkpoint_diversity = tf.train.latest_checkpoint(checkpoint_diversity)
        if checkpoint_asp != '' and gfile.IsDirectory(checkpoint_asp):
            checkpoint_asp = tf.train.latest_checkpoint(checkpoint_asp)
        print("critic checkpoint: %s" % checkpoint_critic)
        print("generator checkpoint: %s" % checkpoint_generator)
        print("actor checkpoint: %s" % checkpoint_actor)
        print("asp checkpoint: %s" % checkpoint_asp)
        print("diversity checkpoint: %s" % checkpoint_diversity)
        print("load critic model...")
        model.critic_saver.restore(sess, checkpoint_critic)
        if model.generator_saver is not None:
            print("load generator model...")
            model.generator_saver.restore(sess, checkpoint_generator)
        if model.asp_saver is not None:
            print("load asp model...")
            model.asp_saver.restore(sess, checkpoint_asp)
        if model.diversity_saver is not None:
            print("load diversity model...")
            model.diversity_saver.restore(sess, checkpoint_diversity)
        if model.actor_saver is not None:
            print("load actor model...")
            model.actor_saver.restore(sess, checkpoint_actor)
        print("load checkpoint done")
        
        saver = tf.train.Saver()
        export_dir = FLAGS.path['export_dir']
        if export_dir.find("hdfs") != -1:
            export_version = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M')
            export_dir = os.path.join(export_dir, export_version, 'data')
        model_path = os.path.join(export_dir, "model.ckpt")
        save_path = saver.save(sess, model_path, global_step=global_step)
        print("Model saved in path: %s" % save_path)
        tf.train.write_graph(sess.graph, export_dir, 'graph.pbtxt')

        save_path = os.path.join(export_dir, "fg.json")
        with tf.afile.AFile(save_path, "w") as f:
            json.dump(fg_configs, f, indent=4)
            print("fg saved in path: %s" % save_path)


def run(FLAGS):
    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    cluster = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    worker_count = len(worker_hosts)
    print('worker count: {}, job name: {}, task index: {}'.format(worker_count, FLAGS.job_name, FLAGS.task_index))
    
    if FLAGS.job_name == 'ps':
        server.join()
    elif FLAGS.job_name == 'worker':
        export(FLAGS)
