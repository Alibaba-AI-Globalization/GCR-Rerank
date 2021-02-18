"""
@author xxx
"""
import featureparser_fg
from layers.layer_lib import build_dnn_net, build_logits, scaled_dot_product_attention
from model_base import ModelBase
import tensorflow as tf
import json
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope as vs
from transforms.data_format import get_json_conf_from_file, format_feature_offline
from transforms.preprocessing import Preprocess
from transforms.to_net0 import to_net0_dict
import importlib
from util import tf_summary
from util.rtm import MetricsRT
from layers.transformer import MutilHeadAttention

is_debug = False


class Model(ModelBase):
    def __init__(self, FLAGS, input_item_num=50, candidate_size=None, slate_item_num=10, global_step=None,
                 batch_size=None, variable_scope=None):
        
        super(Model, self).__init__(name='Generator Model')
        
        self.flags = FLAGS
        self.mrt = MetricsRT(FLAGS)
        self.preprocess_conf = FLAGS.feature_conf['generator_fpre']
        self.model_conf = None
        self.feature_conf = None
        self.label_conf = None
        
        self.feature_dict = None
        self.featureparser_feature_dict = None
        self.item_num = input_item_num
        self.slate_item_num = slate_item_num
        self.topk = slate_item_num
        self.slate_num = 1
        self.pre_index = None
        self.global_step = global_step
        
        self.score_size = 22
        self.price_level_size = 13
        self.embed_size = 8
        
        self.model_conf = FLAGS.generator_model
        with open(FLAGS.feature_conf['generator_features'], 'r') as fp:
            self.feature_conf = json.load(fp)
        with open(FLAGS.feature_conf['label'], 'r') as fp:
            self.label_conf = json.load(fp)
        with open(FLAGS.feature_conf['input'], 'r') as fp:
            conf = json.load(fp)
            feature_side = {}
            for item in conf["features"]:
                feature_side[item['feature_name']] = item['expression'].split(":")[0]
        self.feature_side = feature_side
        
        self.net_arch = self.model_conf['architecture']
        self.price_gru_dimension = self.net_arch['price_gru_dimension']
        self.score_gru_dimension = self.net_arch['score_gru_dimension']
        self.embed_gru_dimension = self.net_arch['embed_gru_dimension']
        self.add_diversity = self.net_arch['add_diversity']
        self.pbc_diversity = self.net_arch['pbc_diversity'] if 'pbc_diversity' in self.net_arch else False
        self.variable_scope = self.net_arch['variable_scope'] if 'variable_scope' in self.net_arch else None
        if variable_scope is not None:
            self.variable_scope = variable_scope
        if self.variable_scope is None:
            self.variable_scope = "generator_network"
        self.use_position = self.net_arch['use_position']
        self.score_type = self.net_arch['score_type']
        self.sample_type = self.net_arch['sample_type']
        self.candidate_size = self.net_arch['use_candidate_size']
        if candidate_size is not None:
            self.candidate_size = candidate_size
        
        self.train_conf = self.model_conf['train']
        self.sample_buffer_size = self.train_conf['sample_buffer_size'] if FLAGS.mode in ['train', "local"] else 1
        if FLAGS.mode == "local" and FLAGS.model == "rerank":
            self.sample_buffer_size = 1
        self.isp_mode = self.train_conf['isp_mode']
        self.critic_score_weight = self.train_conf['critic_score_weight']
        self.reward_type = self.train_conf['reward_type']
        self.target_score = self.train_conf['target_score']
        self.use_ppo = self.train_conf['use_ppo']
        self.user_power_mode = self.train_conf['user_power_mode'] if 'user_power_mode' in self.train_conf else False
        self.generate_loss_type = self.train_conf['generate_loss_type']
        self.batch_size = self.train_conf['batch_size'] if FLAGS.mode != 'export' else 1
        if batch_size is not None:
            self.batch_size = batch_size
        self.loss = None
        self.loss_ema = None
        self.train_op = None
        self.critic_saver = None
        self.need_beam_search = False
        
        self.generator_items = None
        self.generator_score = None
    
    def deal_su_by_mean(self, feature_dict):
        su_list = []
        for feat_name in feature_dict:
            if feat_name.startswith("su_") and not feat_name.endswith("_price_percent"):
                # shape: [B, S, E] => [B, 1, E]
                su_embed = tf.reduce_mean(feature_dict[feat_name], axis=1, keep_dims=True)
                print "  add:", feat_name, su_embed.get_shape()
                su_list.append(su_embed)
            elif feat_name.startswith("su_") and feat_name.endswith("_price_percent"):
                # shape: [B, S] => [B, 1, 1]
                price_percent = tf.expand_dims(feature_dict[feat_name], axis=2)
                price_percent = tf.reduce_mean(price_percent, axis=1, keep_dims=True)
                print "  add:", feat_name, price_percent.get_shape()
                su_list.append(price_percent)
        su_net = tf.concat(su_list, axis=2)
        return su_net

    def position_embedding(self):
        bucket_size = self.candidate_size
        embedding_size = 16
        initializer = None
        with tf.variable_scope(self.variable_scope, reuse=tf.AUTO_REUSE):
            if initializer and initializer.startswith("uniform"):
                val = float(initializer.split(":")[1])
                embeddings = vs.get_variable("item_pos_embedding", [bucket_size, embedding_size],
                                             initializer=tf.random_uniform_initializer(-val, val),
                                             collections=[ops.GraphKeys.GLOBAL_VARIABLES,
                                                          ops.GraphKeys.MODEL_VARIABLES])
            else:
                embeddings = vs.get_variable("item_pos_embedding", [bucket_size, embedding_size],
                                             initializer=tf.random_uniform_initializer(-1.0, 1.0),
                                             collections=[ops.GraphKeys.GLOBAL_VARIABLES,
                                                          ops.GraphKeys.MODEL_VARIABLES])
    
        id_tensor = tf.range(self.item_num)
        embed = tf.nn.embedding_lookup(embeddings, id_tensor)
        embed = tf.tile(embed, [self.batch_size, 1])
        return embed

    def rank_embedding(self, id_tensor, embedding_name, embedding_size, bucket_size):
        embedding_size = embedding_size
        initializer = None
        with tf.variable_scope(self.variable_scope, reuse=tf.AUTO_REUSE):
            if initializer and initializer.startswith("uniform"):
                val = float(initializer.split(":")[1])
                embeddings = vs.get_variable(embedding_name, [bucket_size, embedding_size],
                                             initializer=tf.random_uniform_initializer(-val, val),
                                             collections=[ops.GraphKeys.GLOBAL_VARIABLES,
                                                          ops.GraphKeys.MODEL_VARIABLES])
            else:
                embeddings = vs.get_variable(embedding_name, [bucket_size, embedding_size],
                                             initializer=tf.random_uniform_initializer(-1.0, 1.0),
                                             collections=[ops.GraphKeys.GLOBAL_VARIABLES,
                                                          ops.GraphKeys.MODEL_VARIABLES])
    
        embed = tf.nn.embedding_lookup(embeddings, id_tensor)
        return embed
    
    def add_trick_features(self):
        # add item_pos
        item_pos = self.position_embedding()
        self.feature_dict['item_pos'] = item_pos
        
        # add one_hot item_pos
        pos_tensor = tf.tile(tf.eye(self.candidate_size)[:self.item_num, :], [self.batch_size, 1])
        pos_tensor = tf.reshape(tf.cast(pos_tensor, tf.float32), [-1, self.candidate_size])
        self.feature_dict['item_pos_one_hot'] = pos_tensor
        
        # add price_gap
        if "real_price" in self.feature_dict and "avg_price_7d" in self.feature_dict:
            print "add price_gap"
            real_price = self.feature_dict["real_price"]
            avg_price_7d = self.feature_dict["avg_price_7d"]
            avg_price_7d = tf.where(tf.equal(avg_price_7d, 0), real_price, avg_price_7d)
            avg_price_7d = tf.where(tf.equal(avg_price_7d, 0), tf.ones_like(avg_price_7d, dtype=tf.float32),
                                    avg_price_7d)
            price_gap = tf.clip_by_value(real_price / avg_price_7d - 1, -5, 5)
            self.feature_dict["price_gap"] = tf.reshape(price_gap, [-1, 1])

    def get_net0_dict(self, featureparser_feature_dict):
        with tf.name_scope("generator_feature_pipe"):
            self.featureparser_feature_dict = featureparser_feature_dict
            preprocess = Preprocess(self.preprocess_conf)
            self.feature_dict = preprocess.process(featureparser_feature_dict, list_size=self.item_num,
                                                   variable_scope=self.variable_scope)
            self.add_trick_features()
            net0_dict = to_net0_dict(self.feature_dict, conf=self.feature_conf, list_size=self.candidate_size)
            # features
            price = net0_dict['price']
            item_dense = net0_dict['item_dense']
            context_features = net0_dict['context_features']
            embeddings = net0_dict['embeddings']
            trick = net0_dict['trick'] if 'trick' in net0_dict else None
            
            # user query
            query = tf.reduce_mean(net0_dict['query'], axis=1, keep_dims=True)
            if 'usr_curiosity_vector' in net0_dict:
                usr_curiosity_vector = tf.expand_dims(net0_dict['usr_curiosity_vector'], axis=1)
                query = tf.concat([query, usr_curiosity_vector], axis=2)
            
            score_featues = tf.concat([price, item_dense, context_features], axis=1)
            self.score_size = score_featues.shape[1].value - self.price_level_size
            if trick is not None:
                net0_embed = tf.concat([score_featues, trick, embeddings], axis=1)
            else:
                net0_embed = tf.concat([score_featues, embeddings], axis=1)
            net0_dict['net0_embed'] = net0_embed
            
            su_embed = self.deal_su_by_mean(self.feature_dict)
            su_embed = tf.concat([su_embed], axis=2)
            print "  mean, add su_embed, shape: %s", su_embed.get_shape()
            net0_dict['su_rt'] = su_embed
            tf_summary.add_net0_summary(net0_dict)
        
        return net0_dict
    
    def build_sc_net(self, net0_dict):
        net_arch = self.net_arch
        input_feat = net0_dict['net0_embed']
        gru_input = tf.reshape(input_feat, [self.batch_size, self.item_num, input_feat.shape[1]])
        
        with tf.variable_scope(self.variable_scope + "/sc_net"):
            gru_cell = tf.contrib.rnn.GRUCell(num_units=net_arch['sc_gru_dimension'])
            output, state = tf.nn.dynamic_rnn(gru_cell, inputs=gru_input, dtype=tf.float32, parallel_iterations=128)
            state = tf.reshape(state, [-1, net_arch['sc_gru_dimension']])
            state = tf.reshape(tf.tile(state, [1, self.item_num]), [-1, net_arch['sc_gru_dimension']])
        return state
    
    def build_self_attention_sc_net(self, net0_dict):
        net_arch = self.net_arch
        print "net_arch", net_arch
        input_feat = net0_dict['net0_embed']
        
        with tf.variable_scope(self.variable_scope + "/sc_net"):
            dim = net_arch['transformer_head']
            linear = build_dnn_net(input_feat, name="linear", hidden_units=[dim], activation=[None],
                                   reused=tf.AUTO_REUSE)
            input_seq = tf.reshape(linear, [self.batch_size, self.item_num, linear.shape[1]])
            state, weights = scaled_dot_product_attention(input_seq, input_seq, input_seq)
            state = tf.reduce_mean(state, axis=1, keep_dims=True)
            state = tf.reshape(tf.tile(state, [1, self.item_num, 1]), [-1, dim])
        return state

    def build_transformer_sc_net(self, net0_dict):
        net_arch = self.net_arch
        print "net_arch", net_arch
        input_feat = net0_dict['net0_embed']
        input_seq = tf.reshape(input_feat, [self.batch_size, self.item_num, input_feat.shape[1]])
    
        with tf.variable_scope(self.variable_scope + "/sc_net"):
            dim = net_arch['transformer_dim']
            head = net_arch['transformer_head']
            mutil_head_attention = MutilHeadAttention(dim, head)
            state, weights = mutil_head_attention(input_seq, input_seq, input_seq)
            state = tf.reshape(state, [-1, dim])
        return state
    
    def build_gru_cell(self, dnn_parent_scope):
        with tf.variable_scope(dnn_parent_scope, reuse=tf.AUTO_REUSE):
            row_num = self.batch_size * self.sample_buffer_size * self.slate_num
            next_price_level_input = tf.zeros([row_num, self.price_level_size])
            next_score_input = tf.zeros([row_num, self.score_size])
            next_embed_input = tf.zeros([row_num, self.embed_size])
            with tf.variable_scope('price', reuse=tf.AUTO_REUSE):
                self.price_cell = tf.contrib.rnn.GRUCell(num_units=self.price_gru_dimension)
                price_state_init = self.price_cell.zero_state(row_num, tf.float32)  # [ B * num_units ] start state.
                _, self.price_state = self.price_cell(next_price_level_input, price_state_init)
            with tf.variable_scope('score', reuse=tf.AUTO_REUSE):
                self.score_cell = tf.contrib.rnn.GRUCell(num_units=self.score_gru_dimension)
                score_state_init = self.score_cell.zero_state(row_num, tf.float32)  # [ B * num_units ] start state.
                _, self.score_state = self.score_cell(next_score_input, score_state_init)
            with tf.variable_scope('embed', reuse=tf.AUTO_REUSE):
                self.embed_cell = tf.contrib.rnn.GRUCell(num_units=self.embed_gru_dimension)
                embed_state_init = self.embed_cell.zero_state(row_num, tf.float32)  # [ B * num_units ] start state.
                _, self.embed_state = self.embed_cell(next_embed_input, embed_state_init)
            self.all_state = tf.concat([self.price_state, self.score_state, self.embed_state], axis=1)
            
            if self.add_diversity == 'true':
                batch = self.batch_size * self.sample_buffer_size * self.slate_num
                self.price_w = tf.random_uniform([self.price_level_size, 1], maxval=0.1)
                self.score_w = tf.random_uniform([self.score_size, 1], maxval=0.1)
                self.embed_w = tf.random_uniform([self.embed_size, 1], maxval=0.1)
                self.price_diversity = tf.zeros([batch, 0])  # [B * buffer, 0]
                self.score_diversity = tf.zeros([batch, 0])
                self.embed_diversity = tf.zeros([batch, 0])
                self.price_diversity_state = tf.zeros([batch, 10])  # [B * buffer, 10]
                self.score_diversity_state = tf.zeros([batch, 10])
                self.embed_diversity_state = tf.zeros([batch, 10])
            if self.pbc_diversity:
                self.pbc_diversity_state = None
                self.pbc_diversity_cnt = None
                self.pbc_is_new = None

                brand_id = tf.as_string(self.feature_dict['raw_cate_id_encode'])
                cate_id = tf.as_string(self.feature_dict['raw_brand_id_encode'])
                price_level = tf.as_string(self.feature_dict['raw_price_level'])
                self.bcp_ori = brand_id + '-' + cate_id + '-' + price_level
    
    def build_dynamic_feature(self):
        batchIndex = tf.reshape(tf.range(self.batch_size, dtype=tf.int64), [-1, 1])
        batchIndex = tf.reshape(tf.tile(batchIndex, [1, self.sample_buffer_size * self.slate_num]), [-1, 1])
        
        nextInputIndexBias = batchIndex * tf.cast(self.item_num, tf.int64)
        nextInputIndex = nextInputIndexBias + tf.reshape(self.selectedIndexStop, [-1, 1])
        if is_debug:
            self.nextInputIndex = nextInputIndex
            self.nextInputIndexBias = nextInputIndexBias
        nextInput = tf.gather_nd(self.net0_dict['net0_embed'], nextInputIndex)
        next_price_level_input = nextInput[:, :self.price_level_size]
        next_score_input = nextInput[:, self.price_level_size:self.score_size + self.price_level_size]
        next_embed_input = nextInput[:,
                           self.score_size + self.price_level_size:self.score_size + self.price_level_size + self.embed_size]
        _, self.price_state = self.price_cell(next_price_level_input, self.price_state)
        _, self.score_state = self.score_cell(next_score_input, self.score_state)
        _, self.embed_state = self.embed_cell(next_embed_input, self.embed_state)
        self.all_state = tf.concat([self.price_state, self.score_state, self.embed_state], axis=1)
        if self.add_diversity == 'true':
            price_diversity = tf.matmul(next_price_level_input, self.price_w)  # [B * buffer, 1]
            score_diversity = tf.matmul(next_score_input, self.score_w)
            embed_diversity = tf.matmul(next_embed_input, self.embed_w)
            self.price_diversity = tf.concat([self.price_diversity, price_diversity], axis=1)
            self.score_diversity = tf.concat([self.score_diversity, score_diversity], axis=1)
            self.embed_diversity = tf.concat([self.embed_diversity, embed_diversity], axis=1)
            size = tf.shape(self.price_diversity)[1]
            self.price_diversity_state = tf.pad(self.price_diversity, [[0, 0], [0, 10 - size]])  # [B * buffer, 10]
            self.score_diversity_state = tf.pad(self.score_diversity, [[0, 0], [0, 10 - size]])
            self.embed_diversity_state = tf.pad(self.embed_diversity, [[0, 0], [0, 10 - size]])
        if self.pbc_diversity:
            # calc now diversity
            bcp = tf.reshape(self.bcp_ori, [-1, 1])
            bcp = tf.gather_nd(bcp, nextInputIndex)
            
            if self.pbc_diversity_state is None:
                self.pbc_diversity_state = bcp
                self.bcp_is_new = tf.ones(tf.shape(bcp))
                self.neighbor_same = tf.zeros(tf.shape(bcp))
            else:
                bcp_same = tf.equal(self.pbc_diversity_state, bcp)
                bcp_is_new = 1 - tf.cast(tf.reduce_any(bcp_same, axis=1, keep_dims=True), dtype=tf.float32)
                self.pbc_diversity_state = tf.concat([self.pbc_diversity_state, bcp], axis=1)
                self.bcp_is_new = tf.concat([self.bcp_is_new, bcp_is_new], axis=1)
                neighbor_same = tf.cast(bcp_same[:, -1:], dtype=tf.float32)
                self.neighbor_same = tf.concat([self.neighbor_same, neighbor_same], axis=1)
            
            pbc_diversity_cnt = tf.reduce_sum(self.bcp_is_new, axis=1)
            pbc_diversity_cnt = tf.stop_gradient(pbc_diversity_cnt)
            pbc_diversity_cnt = tf.reshape(tf.cast(pbc_diversity_cnt, dtype=tf.float32), [-1, 1])
            
            if self.pbc_diversity_cnt is None:
                self.pbc_diversity_cnt = pbc_diversity_cnt
            else:
                self.pbc_diversity_cnt = tf.concat([self.pbc_diversity_cnt, pbc_diversity_cnt], axis=1)
    
    def build_score_network(self, dnn_parent_scope):
        if self.use_position:
            # [B, num_units] => [B, num_units*N] = > [B*N, num_units]
            num_units = self.price_gru_dimension + self.score_gru_dimension + self.embed_gru_dimension + 10
            padState = tf.reshape(tf.tile(self.all_state, [1, self.item_num]), [-1, num_units])
        else:
            # [B, num_units] => [B, num_units*N] = > [B*N, num_units]
            num_units = self.price_gru_dimension + self.score_gru_dimension + self.embed_gru_dimension
            padState = tf.reshape(tf.tile(self.all_state, [1, self.item_num]), [-1, num_units])
        
        # build score network net0 input
        net0_feature_num = self.net0_dict['net0_embed'].shape[1]
        net0_input = tf.reshape(
            tf.tile(tf.reshape(self.net0_dict['net0_embed'], [self.batch_size, -1]),
                    [1, self.sample_buffer_size * self.slate_num]),
            [self.batch_size * self.sample_buffer_size * self.slate_num * self.item_num, net0_feature_num])
        selectInput = tf.concat([net0_input, padState], axis=1)
        
        decodeNet = selectInput
        if self.add_diversity == 'true':
            # diversity
            price_diversity_state = tf.reshape(tf.tile(self.price_diversity_state, [1, self.item_num]),
                                               [-1, 10])  # [B * buffer * N, 10]
            score_diversity_state = tf.reshape(tf.tile(self.score_diversity_state, [1, self.item_num]),
                                               [-1, 10])  # [B * buffer * N, 10]
            embed_diversity_state = tf.reshape(tf.tile(self.embed_diversity_state, [1, self.item_num]),
                                               [-1, 10])  # [B * buffer * N, 10]
            diversity_state_list = [price_diversity_state, score_diversity_state, embed_diversity_state]
            # item input
            price_level_input = selectInput[:, :self.price_level_size]
            score_input = selectInput[:, self.price_level_size:self.score_size + self.price_level_size]
            embed_input = selectInput[:,
                          self.score_size + self.price_level_size:self.score_size + self.price_level_size + self.embed_size]
            price_diversity = tf.matmul(price_level_input, self.price_w)  # [B * buffer * N, 1]
            score_diversity = tf.matmul(score_input, self.score_w)
            embed_diversity = tf.matmul(embed_input, self.embed_w)
            price_same = tf.cast(tf.equal(price_diversity_state, price_diversity),
                                 dtype=tf.float32)  # [B * buffer * N, 10]
            score_same = tf.cast(tf.equal(score_diversity_state, score_diversity), dtype=tf.float32)
            embed_same = tf.cast(tf.equal(embed_diversity_state, embed_diversity), dtype=tf.float32)
            topk = tf.cast(self.topk, dtype=tf.float32)
            price_ratio = tf.reduce_sum(price_same, axis=1, keep_dims=True) / topk  # [B * buffer * N, 1]
            score_ratio = tf.reduce_sum(score_same, axis=1, keep_dims=True) / topk
            embed_ratio = tf.reduce_sum(embed_same, axis=1, keep_dims=True) / topk
            price_new = tf.cast(tf.equal(price_ratio, 1 / topk), dtype=tf.float32)  # [B * buffer * N, 1]
            score_new = tf.cast(tf.equal(score_ratio, 1 / topk), dtype=tf.float32)
            embed_new = tf.cast(tf.equal(embed_ratio, 1 / topk), dtype=tf.float32)
            diversity_state_list.extend(
                [price_same, score_same, embed_same, price_ratio, score_ratio, embed_ratio, price_new, score_new,
                 embed_new])
            diversity_state = tf.concat(diversity_state_list, axis=1)
            self.diversity_state_stop = tf.stop_gradient(diversity_state)
            decodeNet = tf.concat([decodeNet, self.diversity_state_stop], axis=1)  # [B * buffer * N, num_features]
        
        if self.pbc_diversity:
            pbc_diversity_state = None
            if self.pbc_diversity_state is not None:
                gen_num = self.pbc_diversity_state.shape[1]
                pbc_diversity_state = tf.reshape(tf.tile(self.pbc_diversity_state, [1, self.item_num]),
                                                 [-1, gen_num])  # [B * buffer * N, 10]
                bcp_is_new_hist = tf.reshape(tf.tile(self.bcp_is_new, [1, self.item_num]),
                                                 [-1, gen_num])  # [B * buffer * N, 10]
                # pbc diversity
                bcp = self.bcp_ori
                # tile buffer
                bcp = tf.reshape(bcp, [-1, self.item_num])
                bcp = tf.reshape(tf.tile(bcp, [1, self.sample_buffer_size]), [-1, 1])
                
                bcp_same = tf.equal(pbc_diversity_state, bcp)
                bcp_is_new = 1 - tf.cast(tf.reduce_any(bcp_same, axis=1, keep_dims=True), dtype=tf.float32)
                bcp_is_new = tf.concat([bcp_is_new_hist, bcp_is_new], axis=1)
                pbc_diversity_cnt = tf.reduce_sum(bcp_is_new, axis=1)
            else:
                pbc_diversity_cnt = tf.ones(shape=[self.batch_size * self.sample_buffer_size, self.item_num, 1])
            
            pbc_diversity_cnt = tf.reshape(tf.cast(pbc_diversity_cnt, dtype=tf.float32), [-1, 1]) / 10
            self.pbc_diversity_cnt_stop = tf.stop_gradient(pbc_diversity_cnt)
            decodeNet = tf.concat([decodeNet, self.pbc_diversity_cnt_stop], axis=1)  # [B * buffer * N, num_features]
        
        num_features = decodeNet.shape[1]
        decodeNet = tf.reshape(decodeNet, [self.batch_size, self.slate_num, -1])
        logitsList = []
        for i in range(self.slate_num):
            inputNet = tf.reshape(decodeNet[:, i, :], [-1, num_features])
            if self.slate_num > 1:
                dnn_parent_scope2 = dnn_parent_scope + "_" + str(i)
            else:
                dnn_parent_scope2 = dnn_parent_scope
            
            with tf.variable_scope(dnn_parent_scope2, reuse=tf.AUTO_REUSE):
                net = build_dnn_net(inputNet, name='score_net',
                                    hidden_units=self.net_arch['generate_units'],
                                    activation=self.net_arch['generate_activation'], reused=tf.AUTO_REUSE)
                
                logits = build_logits(net, name='score_net')
                logitsList.append(tf.reshape(logits, [-1, self.item_num]))  # [batch * buffer, N]
            
            if self.reward_type == "rnd_reward":
                with tf.variable_scope(dnn_parent_scope2, reuse=tf.AUTO_REUSE):
                    rnd_net = build_dnn_net(inputNet, name='rnd_net', hidden_units=[64, 64],
                                            activation=['relu', 'relu'], reused=tf.AUTO_REUSE)
                    rnd_pred = build_logits(rnd_net, name='rnd_net')
                    rnd_net_fix = build_dnn_net(inputNet, name='rnd_net_fix', hidden_units=[64, 64],
                                                activation=['relu', 'relu'], reused=tf.AUTO_REUSE, is_training=False)
                    rnd_label = build_logits(rnd_net_fix, name='rnd_net_fix')
                    rnd_reward = tf.square(rnd_pred - rnd_label)
                    rnd_mean, rnd_variance = tf.nn.moments(tf.reshape(rnd_reward, [-1]), [0])
                    rnd_stddev = tf.sqrt(rnd_variance)
                    rnd_reward = tf.clip_by_value((rnd_reward-rnd_mean)/rnd_stddev, 0, 1)  # [batch * buffer * N, 1]
                    self.rnd_reward = tf.reshape(rnd_reward, [-1, self.item_num])
        
        self.logits = tf.concat(logitsList, axis=1)  # [batch * buffer, slate_num * N]
        self.logits = tf.reshape(self.logits, [-1, self.item_num])  # [Batch * buffer * slate_num, N]
    
    def sample(self, dnn_logits, score, gen_idx=None):
        new_logits = dnn_logits
        if self.flags.mode in ['test', 'export', 'local'] and self.user_power_mode is True:
            print("===> use user_power_mode")
            user_power = tf.reshape(self.feature_dict['user_power'], [-1])[0]
            price_level = self.featureparser_feature_dict['price']
            print(price_level, user_power)
            mask = tf.where(tf.less_equal(price_level, user_power-4)
                            , tf.ones_like(price_level)*20, tf.zeros_like(price_level))
            new_logits = new_logits - tf.cast(mask, dtype=tf.float32)
            selectIndex = tf.cast(tf.reshape(tf.argmax(new_logits, axis=1), [self.batch_size * self.slate_num, 1]),
                                  tf.int64)
        elif self.flags.mode in ['test', 'export', 'local_predict'] or self.sample_type == 'max':
            print("===> use max")
            selectIndex = tf.cast(tf.reshape(tf.argmax(new_logits, axis=1), [self.batch_size * self.slate_num, 1]),
                                  tf.int64)
            if self.isp_mode == 'true':
                selectIndex = tf.reshape(self.pre_index[:, gen_idx], [-1, 1])
        elif self.sample_type == 'softmax':
            selectIndex = tf.multinomial(new_logits, 1)
            selectIndex = tf.reshape(selectIndex, [self.batch_size * self.sample_buffer_size * self.slate_num, 1])
        
        batchIndex = tf.reshape(tf.range(self.batch_size * self.sample_buffer_size * self.slate_num, dtype=tf.int64),
                                [-1, 1])
        fillIndex = tf.concat([batchIndex, selectIndex], axis=1)
        selectScore = tf.gather_nd(score, fillIndex)
        selectScore = tf.reshape(selectScore, [self.batch_size * self.sample_buffer_size * self.slate_num, 1])
        return selectIndex, selectScore
    
    def build_network(self, net0_dict):
        net_arch = self.model_conf['architecture']
        # 1. encode top 100 as encode network
        if net_arch['sc_gru_dimension'] > 0:
            if net_arch['sc_encoder'] == "transformer":
                sc = self.build_transformer_sc_net(net0_dict)
            elif net_arch['sc_encoder'] == "self_attention":
                sc = self.build_self_attention_sc_net(net0_dict)
            elif net_arch['sc_encoder'] == "gru":
                sc = self.build_sc_net(net0_dict)
            else:
                sc = self.build_sc_net(net0_dict)
            net0_dict['net0_embed'] = tf.concat([net0_dict['net0_embed'], sc], axis=1)
        
        if "su_rt" in net0_dict:
            # user realtime features
            print "add su_rt to encode network"
            su_rt = tf.reshape(tf.tile(net0_dict["su_rt"], [1, self.item_num, 1]),
                               [-1, net0_dict["su_rt"].shape[2]])
            net0_dict['net0_embed'] = tf.concat([net0_dict['net0_embed'], su_rt], axis=1)
        self.net0_dict = net0_dict
        
        index_state = tf.TensorArray(size=0, dtype=tf.int64, dynamic_size=True,
                                     clear_after_read=False)  # , element_shape=[self.batch_size, 1])
        score_state = tf.TensorArray(size=0, dtype=tf.float32, dynamic_size=True,
                                     clear_after_read=False)  # , element_shape=[self.batch_size, 1])
        batch_index = tf.reshape(
            tf.range(self.batch_size * self.sample_buffer_size * self.slate_num, dtype=tf.int64),
            [-1, 1])
        
        for i in range(10):
            if i == 0:
                self.build_gru_cell(self.variable_scope + "/sg_net")
            else:
                self.build_dynamic_feature()
                if is_debug and i == 3:
                    self.i3_nextInputIndex = self.nextInputIndex
                    self.i3_nextInputIndexBias = self.nextInputIndexBias
            
            if self.use_position:
                print('add position state')
                one_hot = [0.0 for j in range(10)]
                one_hot[i] = 1.0
                self.all_state = tf.concat(
                    [self.all_state, tf.tile(tf.reshape(tf.convert_to_tensor(one_hot), [1, -1]),
                                             [self.batch_size * self.sample_buffer_size * self.slate_num, 1])], axis=1)
            self.build_score_network(self.variable_scope + "/score_net")
            
            self.logits = tf.clip_by_value(self.logits, -10, 10)
            
            dnn_exp = tf.exp(self.logits)
            softmax = dnn_exp / tf.reduce_sum(dnn_exp, axis=1, keep_dims=True)
            if i == 0:
                self.entropy = tf.reduce_mean(
                    -tf.reduce_sum(softmax * tf.log(tf.clip_by_value(softmax, 1e-10, 1.0)), axis=1))
            else:
                self.entropy += tf.reduce_mean(
                    -tf.reduce_sum(softmax * tf.log(tf.clip_by_value(softmax, 1e-10, 1.0)), axis=1))
            
            if self.score_type == 'tanh':
                self.decodeScore = tf.tanh(self.logits)
            elif self.score_type == 'sigmoid':
                self.decodeScore = tf.sigmoid(self.logits)
            elif self.score_type == 'softmax':
                self.decodeScore = softmax
            elif self.score_type == 'logits':
                self.decodeScore = self.logits
            
            if i > 0:
                selectedIndex = tf.reshape(index_state.stack(), [-1, 1])
                sparseIndexStop = tf.stop_gradient(tf.cast(selectedIndex, tf.int64))
                tileBatchIndex = tf.tile(batch_index, [i, 1])
                fillIndex = tf.concat([tileBatchIndex, sparseIndexStop], axis=1)
                # fillIndex column1: batch index column2: selected index
                maskValues = tf.tile([80.0], [tf.shape(fillIndex)[0]])
                mask = tf.SparseTensor(fillIndex, maskValues, tf.shape(self.logits, out_type=tf.int64))
                self.logits = tf.subtract(self.logits,
                                          tf.sparse_tensor_to_dense(mask, default_value=0.0,
                                                                    validate_indices=False))
                if is_debug and i == 3:
                    self.i3_selectedIndex = sparseIndexStop
                    self.i3_tileBatchIndex = tileBatchIndex
                    self.i3_fillIndex = fillIndex
                    self.i3_logits = self.logits
                    self.i3_hahalfRuleIndex = self.halfRuleIndex
            
            self.selectedIndex, self.selectedScore = self.sample(self.logits, self.decodeScore, i)
            self.selectedIndexStop = tf.stop_gradient(self.selectedIndex)
            index_state = index_state.write(i, self.selectedIndexStop)
            score_state = score_state.write(i, self.selectedScore)
        
        finalIndexState = index_state.stack()  # [K, B*buffer_size, 1]
        finalScoreState = score_state.stack()
        self._selectK = tf.shape(finalIndexState)[0]
        self.finalLoopState = tf.reshape(finalIndexState,
                                         [self._selectK,
                                          self.batch_size * self.sample_buffer_size * self.slate_num])
        self.finalLoopState = tf.transpose(self.finalLoopState, [1, 0])
        self.finalScoreState = tf.reshape(finalScoreState,
                                          [self._selectK,
                                           self.batch_size * self.sample_buffer_size * self.slate_num])
        self.finalScoreState = tf.transpose(self.finalScoreState, [1, 0])
        # self._selectK = tf.subtract(self._selectK, 1)
        print('self.finalLoopState get_shape() = %s' % self.finalLoopState.get_shape())
        print('self.finalScoreState get_shape() = %s' % self.finalScoreState.get_shape())
        if self.batch_size == 1:
            self.finalLoopState = tf.cond(self.item_num <= 10,
                                          lambda: tf.cast(tf.reshape(tf.range(0, self.item_num), [1, -1]),
                                                          tf.int64),
                                          lambda: self.finalLoopState)
        self.generator_items = self.finalLoopState
        self.generator_score = self.finalScoreState
        return self.generator_items
    
    def get_var_dict(self):
        slate_num = self.slate_num
        var_dict = {
            'originScoreSum_10': tf.Variable(tf.zeros([10]), name='metrics_origin_score_10', trainable=False),
            'originScoreSum': tf.Variable(0.0, name='metrics_origin_score', trainable=False),
            'higherSlateSum': tf.Variable(tf.zeros([slate_num]), name='metrics_slate_sum', trainable=False),
            # 'higherScoreSum': tf.Variable(tf.zeros([slate_num]), name='metrics_higher_score_sum', trainable=False),
            # 'lowerScoreSum': tf.Variable(tf.zeros([slate_num]), name='metrics_lower_score_sum', trainable=False),
            'diffScoreSum': tf.Variable(tf.zeros([slate_num]), name='metrics_diff_score_sum', trainable=False),
            'replaceItemSum': tf.Variable(tf.zeros([slate_num]), name='metrics_replace_item_sum', trainable=False),
            'diffAov': tf.Variable(tf.zeros([slate_num]), name='metrics_diff_aov_sum', trainable=False),
            'originPriceSum': tf.Variable(tf.zeros([slate_num]), name='metrics_origin_price_sum', trainable=False),
            'genPriceSum': tf.Variable(tf.zeros([slate_num]), name='metrics_gen_price_sum', trainable=False),
            # cr > 0 & price > 0
            'crHigherSum': tf.Variable(tf.zeros([slate_num]), name='metrics_new_higher_sum', trainable=False),
            'crPriceSum': tf.Variable(tf.zeros([slate_num]), name='metrics_cr_price_sum', trainable=False),
            'cr_cr_gap_sum': tf.Variable(tf.zeros([slate_num]), name='metrics_cr_cr_gap_sum', trainable=False),
            'cr_asp_gap_sum': tf.Variable(tf.zeros([slate_num]), name='metrics_cr_asp_gap_sum', trainable=False),
            # gmv > 0 & price > 0
            'gmv_cr_gap_sum': tf.Variable(tf.zeros([slate_num]), name='metrics_gmv_cr_gap_sum', trainable=False),
            'gmv_asp_gap_sum': tf.Variable(tf.zeros([slate_num]), name='metrics_gmv_asp_gap_sum', trainable=False),
            'gmvHigherSum': tf.Variable(tf.zeros([slate_num]), name='metrics_gmv_higher_sum', trainable=False),
            'gmvPriceSum': tf.Variable(tf.zeros([slate_num]), name='metrics_gmv_price_sum', trainable=False),
            # price > 0
            'priceHigherNum': tf.Variable(tf.zeros([slate_num]), name='metrics_price_higher_sum', trainable=False),
            'pricePriceSum': tf.Variable(tf.zeros([slate_num]), name='metrics_price_price_sum', trainable=False),
            'price_cr_gap_sum': tf.Variable(tf.zeros([slate_num]), name='metrics_price_cr_gap_sum', trainable=False),
            'price_asp_gap_sum': tf.Variable(tf.zeros([slate_num]), name='metrics_price_asp_gap_sum', trainable=False),
            # diversity
            'ori_diversity_sum': tf.Variable(0., name='metrics_ori_diversity_sum', trainable=False),
            'gen_diversity_sum': tf.Variable(0., name='metrics_gen_diversity_sum', trainable=False),
            'ori_diversity_dist': tf.Variable(tf.zeros([10]), name='metrics_ori_diversity_dist', trainable=False),
            'gen_diversity_dist': tf.Variable(tf.zeros([10]), name='metrics_gen_diversity_dist', trainable=False),
            # relevance
            'ori_cate_level_score': tf.Variable(0., name='metrics_ori_cate_level_score_sum', trainable=False),
            'gen_cate_level_score': tf.Variable(0., name='metrics_gen_cate_level_score_sum', trainable=False),
            'ori_recall_by_ori': tf.Variable(0., name='metrics_ori_recall_by_ori_sum', trainable=False),
            'gen_recall_by_ori': tf.Variable(0., name='metrics_gen_recall_by_ori_sum', trainable=False),
        }
        return var_dict
    
    def calc_metrics(self, var_dict, feature_dict):
        CriticModel = importlib.import_module('models.%s' % self.flags.critic_model['name'])
        critic = CriticModel.Model(self.flags, self.slate_item_num)
        
        # 1. fetch feature_dict as selected indexes
        if self.need_beam_search:
            pass
        else:
            select_index = self.generator_items
        self.select_index = select_index
        batch_index = tf.reshape(tf.range(self.batch_size, dtype=tf.int64), [-1, 1, 1])  # [B, 1, 1]
        batch_index = tf.reshape(tf.tile(batch_index, [1, self.slate_num, 1]), [-1, 1, 1])  # [B*slate_num, 1, 1]
        batch_index = tf.tile(batch_index, [1, self.topk, 1])  # [B*slate_num, K, 1]
        select_index = tf.expand_dims(select_index, axis=2)
        # 2. get origin top10 feature_dict
        origin_index = tf.tile(tf.reshape(tf.range(self.topk), [1, -1]),
                               [self.batch_size * self.slate_num, 1])  # [B*slate_num, K] ~ [0,...,9]
        origin_index = tf.expand_dims(tf.cast(origin_index, tf.int64), axis=2)  # [B*slate_num, K, 1]
        self._originFeatureIndex = tf.concat([batch_index, origin_index], axis=2)
        self._criticfeatureIndex = tf.concat([batch_index, select_index], axis=2)
        self._criticFeatureDict = {}
        self._originFeatureDict = {}
        criticFeatureDict = {}
        for name, tensor in feature_dict.items():
            if self.feature_side[name] == 'user':
                self._criticFeatureDict[name] = tf.reshape(tf.tile(tensor, [1, self.slate_num]),
                                                           [self.batch_size * self.slate_num,
                                                            tensor.shape[1]])  # [B, N] => [B*slate_num, N]
                self._originFeatureDict[name] = tf.reshape(tf.tile(tensor, [1, self.slate_num]),
                                                           [self.batch_size * self.slate_num,
                                                            tensor.shape[1]])  # [B, N] => [B*slate_num, N]
            else:
                self._criticFeatureDict[name] = tf.gather_nd(tensor, self._criticfeatureIndex)
                self._originFeatureDict[name] = tf.gather_nd(tensor, self._originFeatureIndex)
            # 3. merge two features into one feature dict; [[generateBatch, K], [originBatch, K]] axis=0
            criticFeatureDict[name] = tf.concat([self._criticFeatureDict[name], self._originFeatureDict[name]],
                                                axis=0)  # [2*B, K]
        # 2. build critic model and get score
        net0_dict = critic.get_net0_dict(criticFeatureDict)
        criticScore = critic.build_network(net0_dict)  # [2*B*10(K), 1]
        self._criticScore = tf.reshape(criticScore, [2 * self.batch_size * self.slate_num, self.topk])  # [2*B,K]
        self.originScore = tf.reduce_mean(tf.slice(self._criticScore, [self.batch_size * self.slate_num, 0], [-1, -1]),
                                          axis=0)
        self.originScore10Add = tf.assign_add(var_dict['originScoreSum_10'], self.originScore, use_locking=True)
        # 3. calculate loss of topK critic score
        if self.target_score == "log_add":
            self.criticLoss = -1.0 * tf.reshape(tf.reduce_sum(tf.log(tf.clip_by_value(1.0 - self._criticScore,
                                                                                      1e-10, 1.)), axis=1), [-1, 1])
        else:
            self.criticLoss = tf.reshape(1.0 - tf.reduce_prod(1.0 - self._criticScore, axis=1), [-1, 1])  # [2*B,1]
        # calc metrics
        self.itemPayGmv = feature_dict['item_pay_gmv']
        genItemPayGmv = tf.exp(tf.gather_nd(self.itemPayGmv, self._criticfeatureIndex))
        originItemPayGmv = tf.exp(tf.gather_nd(self.itemPayGmv, self._originFeatureIndex))
        genItemPayGmv = tf.transpose(tf.reshape(genItemPayGmv, [self.batch_size, self.slate_num, self.topk]),
                                     perm=[1, 0, 2])
        originItemPayGmv = tf.transpose(tf.reshape(originItemPayGmv, [self.batch_size, self.slate_num, self.topk]),
                                        perm=[1, 0, 2])
        
        # self.diffAov = tf.reduce_sum(genItemPayGmv) / tf.reduce_sum(originItemPayGmv)
        slateDiffAov = tf.reduce_sum(genItemPayGmv, axis=2) / tf.reduce_sum(originItemPayGmv, axis=2)  # [slate_num, B]
        
        generateScore = tf.slice(self.criticLoss, [0, 0], [self.batch_size * self.slate_num, 1])
        originScore = tf.slice(self.criticLoss, [self.batch_size * self.slate_num, 0], [-1, 1])
        generateScore = tf.reshape(generateScore, [self.batch_size, self.slate_num])
        originScore = tf.reshape(originScore, [self.batch_size, self.slate_num])
        generateScore = tf.transpose(generateScore, perm=[1, 0])
        originScore = tf.transpose(originScore, perm=[1, 0])
        cr_gap = generateScore / originScore - 1
        asp_gap = slateDiffAov - 1
        gmv_gap = cr_gap + asp_gap
        self.avgGenerateScore = tf.reduce_mean(generateScore, axis=1)
        self.avgOriginScore = tf.reduce_mean(originScore)
        self.originScoreAdd = tf.assign_add(var_dict['originScoreSum'], self.avgOriginScore, use_locking=True)
        diffScore_ = generateScore - originScore
        self.diffScore = tf.reduce_mean(generateScore - originScore, axis=1)
        self.diffScoreAdd = tf.assign_add(var_dict['diffScoreSum'], self.diffScore, use_locking=True)
        higherSum = tf.reduce_sum(tf.cast(tf.greater(diffScore_, 0), tf.float32), axis=1)
        self.higherSumAdd = tf.assign_add(var_dict['higherSlateSum'], higherSum, use_locking=True)
        self.higherRatio = higherSum / self.batch_size
        selectIndex = tf.reshape(select_index, [self.batch_size, self.slate_num, self.topk])
        selectIndex = tf.reshape(tf.transpose(selectIndex, perm=[1, 0, 2]), [self.slate_num, -1])
        replaceIndexSum = tf.reduce_sum(tf.cast(tf.greater(selectIndex, 9), tf.float32), axis=1)
        self.replaceItemSum = tf.assign_add(var_dict['replaceItemSum'], replaceIndexSum, use_locking=True)
        
        genGmv = tf.reduce_sum(genItemPayGmv, axis=2)  # [slate_num, batch_size]
        originGmv = tf.reduce_sum(originItemPayGmv, axis=2)
        # price>0 & cr_gap>0
        crHigherNum = tf.reduce_sum(tf.cast(tf.logical_and(tf.greater(cr_gap, 0), tf.greater(asp_gap, 0)), tf.float32),
                                    axis=1)
        self.crHigherNum = tf.assign_add(var_dict['crHigherSum'], crHigherNum, use_locking=True)
        cr_slate_gmv = tf.reduce_sum(
            tf.where(tf.logical_and(tf.greater(cr_gap, 0), tf.greater(asp_gap, 0)), genGmv, originGmv), axis=1)
        self.crPriceSum = tf.assign_add(var_dict['crPriceSum'], cr_slate_gmv, use_locking=True)
        cr_cr_gap_sum = tf.reduce_sum(tf.where(tf.logical_and(tf.greater(cr_gap, 0), tf.greater(asp_gap, 0)),
                                               cr_gap, tf.zeros_like(cr_gap)), axis=1)
        self.cr_cr_gap_sum = tf.assign_add(var_dict['cr_cr_gap_sum'], cr_cr_gap_sum, use_locking=True)
        cr_asp_gap_sum = tf.reduce_sum(tf.where(tf.logical_and(tf.greater(cr_gap, 0), tf.greater(asp_gap, 0)),
                                                asp_gap, tf.zeros_like(asp_gap)), axis=1)
        self.cr_asp_gap_sum = tf.assign_add(var_dict['cr_asp_gap_sum'], cr_asp_gap_sum, use_locking=True)
        # price>0 & gmv_gap>0
        gmvHigherNum = tf.reduce_sum(
            tf.cast(tf.logical_and(tf.greater(gmv_gap, 0), tf.greater(asp_gap, 0)), tf.float32), axis=1)
        self.gmvHigherSumAdd = tf.assign_add(var_dict['gmvHigherSum'], gmvHigherNum, use_locking=True)
        gmv_slate_price = tf.reduce_sum(tf.where(tf.logical_and(tf.greater(gmv_gap, 0), tf.greater(asp_gap, 0)),
                                                 genGmv, originGmv), axis=1)
        self.gmvPriceSum = tf.assign_add(var_dict['gmvPriceSum'], gmv_slate_price, use_locking=True)
        gmv_cr_gap_sum = tf.reduce_sum(tf.where(tf.logical_and(tf.greater(gmv_gap, 0), tf.greater(asp_gap, 0)),
                                                cr_gap, tf.zeros_like(cr_gap)), axis=1)
        self.gmv_cr_gap_sum = tf.assign_add(var_dict['gmv_cr_gap_sum'], gmv_cr_gap_sum, use_locking=True)
        gmv_asp_gap_sum = tf.reduce_sum(tf.where(tf.logical_and(tf.greater(gmv_gap, 0), tf.greater(asp_gap, 0)),
                                                 asp_gap, tf.zeros_like(asp_gap)), axis=1)
        self.gmv_asp_gap_sum = tf.assign_add(var_dict['gmv_asp_gap_sum'], gmv_asp_gap_sum, use_locking=True)
        # price > 0
        priceHigherNum = tf.reduce_sum(tf.cast(tf.greater(asp_gap, 0), tf.float32), axis=1)
        self.priceHigherNum = tf.assign_add(var_dict['priceHigherNum'], priceHigherNum, use_locking=True)
        price_slate_price = tf.reduce_sum(tf.where(tf.greater(asp_gap, 0), genGmv, originGmv), axis=1)
        self.price_slate_price = tf.assign_add(var_dict['pricePriceSum'], price_slate_price, use_locking=True)
        price_cr_gap_sum = tf.reduce_sum(tf.where(tf.greater(asp_gap, 0), cr_gap, tf.zeros_like(cr_gap)), axis=1)
        self.price_cr_gap_sum = tf.assign_add(var_dict['price_cr_gap_sum'], price_cr_gap_sum, use_locking=True)
        price_asp_gap_sum = tf.reduce_sum(tf.where(tf.greater(asp_gap, 0), asp_gap, tf.zeros_like(asp_gap)), axis=1)
        self.price_asp_gap_sum = tf.assign_add(var_dict['price_asp_gap_sum'], price_asp_gap_sum, use_locking=True)
        # overview
        originItemPayGmv = tf.reshape(originItemPayGmv, [self.slate_num, -1])
        genItemPayGmv = tf.reshape(genItemPayGmv, [self.slate_num, -1])
        self.originPriceSum = tf.assign_add(var_dict['originPriceSum'], tf.reduce_sum(originItemPayGmv, axis=1),
                                            use_locking=True)
        self.genPriceSum = tf.assign_add(var_dict['genPriceSum'], tf.reduce_sum(genItemPayGmv, axis=1),
                                         use_locking=True)
        # calc diversity
        if 'brand_id' in feature_dict:
            brand_id = tf.as_string(feature_dict['brand_id'])
            cate_id = tf.as_string(feature_dict['cate_id'])
            price_level = tf.as_string(feature_dict['item_price_level'])
            bcp = brand_id + '-' + cate_id + '-' + price_level
        else:
            bcp = tf.ones_like(feature_dict['item_pay_gmv'])
        
        gen_bcp = tf.gather_nd(bcp, self._criticfeatureIndex)
        ori_bcp = tf.gather_nd(bcp, self._originFeatureIndex)
        gen_bcp = tf.reshape(gen_bcp, [self.batch_size, self.topk])
        ori_bcp = tf.reshape(ori_bcp, [self.batch_size, self.topk])
        import numpy as np
        def calc_diversity(x):
            x = x.tolist()
            diversity = []
            for i in range(len(x)):
                cnt = len(set(x[i]))
                diversity.append(float(cnt))
            diversity = np.array(diversity, dtype=np.float32)
            return diversity
        
        gen_diversity = tf.py_func(calc_diversity, [gen_bcp], [tf.float32])  # [B, 1]
        ori_diversity = tf.py_func(calc_diversity, [ori_bcp], [tf.float32])
        self.ori_diversity_sum = tf.assign_add(var_dict['ori_diversity_sum'], tf.reduce_sum(ori_diversity), use_locking=True)
        self.gen_diversity_sum = tf.assign_add(var_dict['gen_diversity_sum'], tf.reduce_sum(gen_diversity), use_locking=True)
        
        def to_one_hot(cnt, size=10):
            id_feature = tf.cast(tf.reshape(cnt - 1, [-1, 1]), dtype=tf.int32)
            dim0 = tf.shape(cnt)[0]
            indices = tf.concat([tf.expand_dims(tf.range(0, dim0), 1), id_feature], axis=1)
            id_one_hot = tf.sparse_to_dense(indices, tf.stack([dim0, size]), 1.0, 0.0)
            return id_one_hot

        gen_diver_dist = to_one_hot(tf.reshape(gen_diversity, [-1, 1]))  # [B, K]
        ori_diver_dist = to_one_hot(tf.reshape(ori_diversity, [-1, 1]))
        self.gen_diver_dist_sum = tf.assign_add(var_dict['gen_diversity_dist'],
                                                tf.reduce_sum(gen_diver_dist, axis=0), use_locking=True)
        self.ori_diver_dist_sum = tf.assign_add(var_dict['ori_diversity_dist'],
                                                tf.reduce_sum(ori_diver_dist, axis=0), use_locking=True)
        self.gen_diver_dist_ratio = self.gen_diver_dist_sum / tf.reduce_sum(self.gen_diver_dist_sum)
        self.ori_diver_dist_ratio = self.ori_diver_dist_sum / tf.reduce_sum(self.ori_diver_dist_sum)
        
        metrics = {
            "originScoreSum": self.originScoreAdd,
            "originScoreSum_10": self.originScore10Add,
            "diffScoreSum": self.diffScoreAdd,
            "higherSum": self.higherSumAdd,
            "replaceItemSum": self.replaceItemSum,
            "originPriceSum": self.originPriceSum,
            "genPriceSum": self.genPriceSum,
            # diversity
            "ori_diversity_sum": self.ori_diversity_sum,
            "gen_diversity_sum": self.gen_diversity_sum,
            "ori_diver_dist_sum": self.ori_diver_dist_sum,
            "gen_diver_dist_sum": self.gen_diver_dist_sum,
            "ori_diver_dist_ratio": self.ori_diver_dist_ratio,
            "gen_diver_dist_ratio": self.gen_diver_dist_ratio,
            # cr
            "crHigherNum": self.crHigherNum,
            "crPriceSum": self.crPriceSum,
            "cr_cr_gap_sum": self.cr_cr_gap_sum,
            "cr_asp_gap_sum": self.cr_asp_gap_sum,
            # gmv
            "gmvHigherSum": self.gmvHigherSumAdd,
            "gmvPriceSum": self.gmvPriceSum,
            "gmv_cr_gap_sum": self.gmv_cr_gap_sum,
            "gmv_asp_gap_sum": self.gmv_asp_gap_sum,
            # price > 0
            "priceHigherNum": self.priceHigherNum,
            "pricePriceSum": self.price_slate_price,
            "price_cr_gap_sum": self.price_cr_gap_sum,
            "price_asp_gap_sum": self.price_asp_gap_sum,
        }
        
        return metrics
    
    def build_loss(self, feature_dict):
        CriticModel = importlib.import_module('models.%s' % self.flags.critic_model['name'])
        critic = CriticModel.Model(self.flags, self.slate_item_num)
        
        # 1. fetch feature_dict as selected indexes
        batch_index = tf.reshape(tf.range(self.batch_size, dtype=tf.int64), [-1, 1])  # [B, 1]
        batch_index = tf.reshape(tf.tile(batch_index, [1, self.sample_buffer_size * self.slate_num]),
                                 [-1, 1, 1])  # [B * bufferSize, 1, 1]
        # selectIndex = self.finalLoopState       # [B, K]
        select_index = tf.stop_gradient(self.generator_items)
        select_label_index = select_index
        batch_index = tf.tile(batch_index, [1, self._selectK, 1])  # [B, K, 1]
        select_index = tf.expand_dims(select_index, axis=2)  # [B, K, 1]
        origin_index = tf.tile(tf.reshape(tf.range(self.topk), [1, -1]),
                               [self.batch_size * self.sample_buffer_size * self.slate_num, 1])  # [B, K] ~ [0,...,9]
        origin_index = tf.expand_dims(tf.cast(origin_index, tf.int64), axis=2)
        
        self._originFeatureIndex = tf.concat([batch_index, origin_index], axis=2)
        self._criticfeatureIndex = tf.concat([batch_index, select_index], axis=2)
        self._criticFeatureDict = {}
        self._originFeatureDict = {}
        criticFeatureDict = {}
        for name, tensor in feature_dict.items():
            if not self.feature_side.has_key(name):
                continue
            if self.feature_side[name] == 'user':
                self._criticFeatureDict[name] = tf.reshape(
                    tf.tile(tensor, [1, self.sample_buffer_size * self.slate_num]),
                    [self.batch_size * self.sample_buffer_size * self.slate_num, tensor.shape[1]])
                self._originFeatureDict[name] = tf.reshape(
                    tf.tile(tensor, [1, self.sample_buffer_size * self.slate_num]),
                    [self.batch_size * self.sample_buffer_size * self.slate_num, tensor.shape[1]])
            else:
                self._criticFeatureDict[name] = tf.gather_nd(tensor, self._criticfeatureIndex)
                self._originFeatureDict[name] = tf.gather_nd(tensor, self._originFeatureIndex)
            # 3. merge two features into one feature dict; [[generateBatch, K], [originBatch, K]] axis=0
            criticFeatureDict[name] = tf.concat([self._criticFeatureDict[name], self._originFeatureDict[name]],
                                                axis=0)  # [2*B, K]
        
        # 2. build critic model and get score
        net0_dict = critic.get_net0_dict(criticFeatureDict)
        criticScore = critic.build_network(net0_dict)  # [2*B*10(K), 1]
        self._criticScore = tf.reshape(criticScore, [2 * self.batch_size * self.sample_buffer_size * self.slate_num,
                                                     self.topk])  # [2*B,K]
        
        # 3. calculate loss of topK critic score
        score_total = -1.0 * tf.reshape(
            tf.reduce_sum(tf.log(tf.clip_by_value(1.0 - self._criticScore, 1e-10, 1.0)), axis=1), [-1, 1])
        generateScore_total = tf.slice(score_total, [0, 0],
                                       [self.batch_size * self.sample_buffer_size * self.slate_num, 1])  # [B, 1]
        originScore_total = tf.slice(score_total, [self.batch_size * self.sample_buffer_size * self.slate_num, 0],
                                     [-1, 1])  # [B, 1]
        generateScore_split = tf.slice(self._criticScore, [0, 0],
                                       [self.batch_size * self.sample_buffer_size * self.slate_num, -1])  # [B, K]
        
        # generate multi slate
        if self.slate_num > 1:
            critic_score_weight_list = self.critic_score_weight_list
            critic_score_weight = tf.reshape(tf.constant(critic_score_weight_list), [self.slate_num, 1])
            critic_score_weight = tf.reshape(tf.tile(critic_score_weight, [1, self.topk]),
                                             [1, self.slate_num, self.topk])
        else:
            critic_score_weight = self.critic_score_weight
        
        print("reward_type: {}".format(self.reward_type))
        # asp model reward
        if self.reward_type == "cr_add_gmv":
            self.itemPayGmv = feature_dict['item_pay_gmv']
            genItemPayGmv = tf.gather_nd(self.itemPayGmv, self._criticfeatureIndex)
            genItemPayGmv = tf.clip_by_value(genItemPayGmv, 0.01, 6) / 6.
            genItemPayGmv2 = tf.reshape(genItemPayGmv, [-1, self.slate_num, self.topk])
            generateScore_split2 = tf.reshape(generateScore_split, [-1, self.slate_num, self.topk])
            generateScore_split = generateScore_split2 * critic_score_weight + genItemPayGmv2 * (
                    1 - critic_score_weight)
            generateScore_split = tf.reshape(generateScore_split, [-1, self.topk])
            generateScore_split = generateScore_split * self.critic_score_weight + genItemPayGmv * (
                    1 - self.critic_score_weight)
        elif self.reward_type == "user_power_gmv":
            self.itemPayGmv = feature_dict['item_pay_gmv']
            genItemPayGmv = tf.gather_nd(self.itemPayGmv, self._criticfeatureIndex)
            genItemPayGmv = tf.clip_by_value(genItemPayGmv, 0.01, 6) / 6.

            user_power = tf.reshape(self.feature_dict['user_power'], [-1])[0]
            price_weight = tf.where(tf.logical_and(user_power >= 6, user_power <= 7), user_power*0.05-0.15, 0.)
            generateScore_split = generateScore_split * (1 - price_weight) + genItemPayGmv * price_weight
        elif self.reward_type == "add_real_reward":
            print("reward_type=add_real_reward")
            click = tf.gather_nd(self.label_dict['click'], self._criticfeatureIndex)
            atc = tf.gather_nd(self.label_dict['atc'], self._criticfeatureIndex)
            pay = tf.gather_nd(self.label_dict['pay'], self._criticfeatureIndex)
            generateScore_split = generateScore_split + click*0.3 + atc*0.3 + pay*0.3
        # diversity reward
        elif self.reward_type == "cr_add_diversity":
            import numpy as np
            brand_id = tf.as_string(feature_dict['brand_id'])
            cate_id = tf.as_string(feature_dict['cate_id'])
            price_level = tf.as_string(feature_dict['item_price_level'])
            bcp = brand_id + '-' + cate_id + '-' + price_level
            bcp = tf.gather_nd(bcp, self._criticfeatureIndex)
            
            def calc_diversity(x):
                x = x.tolist()
                diversity = []
                for i in range(len(x)):
                    cnt = len(set(x[i]))
                    diversity.append(float(cnt))
                diversity = np.array(diversity, dtype=np.float32) * 0.1
                return diversity
            
            diver_reward = tf.py_func(calc_diversity, [bcp], [tf.float32])  # [B, 1]
            tf.summary.scalar("diversity_reward", tf.reduce_mean(diver_reward))
            if self.train_conf['use_neg_diversity']:
                diver_reward = tf.reshape(diver_reward, [-1, 1])
                neg_diver_reward = diver_reward - 0.3
                diver_reward = tf.where(diver_reward <= 0.2, neg_diver_reward, tf.zeros_like(diver_reward))
            diver_reward = tf.tile(tf.reshape(diver_reward, [-1, 1]), [1, self.topk])  # [B, k]
            diversity_weight = self.train_conf['diversity_weight']
            generateScore_split = generateScore_split + diver_reward * diversity_weight
        elif self.reward_type == "cumsum":
            generateScore_split = tf.cumsum(generateScore_split, axis=1, reverse=True)
        elif self.reward_type == 'cumsum_mean_std':
            generateScore_split = tf.cumsum(generateScore_split, axis=1, reverse=True)
            gen_score = tf.reshape(generateScore_split, [self.batch_size, self.sample_buffer_size,
                                                         self.slate_num, -1])
            gen_score_mean = tf.reduce_mean(gen_score, axis=1, keep_dims=True)
            square = tf.square(gen_score - gen_score_mean)
            gen_score_std = tf.sqrt(tf.reduce_sum(square, axis=1, keep_dims=True) / (tf.cast((self.topk - 1), tf.float32) + 1e-9))
            generateScore_split = (gen_score - gen_score_mean) / gen_score_std * 10
            generateScore_split = tf.reshape(generateScore_split,
                                             [self.batch_size * self.sample_buffer_size * self.slate_num, -1])
        elif self.reward_type == 'subtract_mean_std':
            gen_score = tf.reshape(generateScore_split, [self.batch_size, self.sample_buffer_size,
                                                         self.slate_num, -1])
            gen_score_mean = tf.reduce_mean(gen_score, axis=1, keep_dims=True)
            square = tf.square(gen_score - gen_score_mean)
            gen_score_std = tf.sqrt(
                tf.reduce_sum(square, axis=1, keep_dims=True) / (tf.cast((self.topk - 1), tf.float32) + 1e-9))
            generateScore_split = (gen_score - gen_score_mean) / gen_score_std * 10
            generateScore_split = tf.reshape(generateScore_split,
                                             [self.batch_size * self.sample_buffer_size * self.slate_num, -1])
        elif self.reward_type == "rnd_reward":
            rnd_reward = tf.gather_nd(self.rnd_reward, self._criticfeatureIndex)
            generateScore_split = generateScore_split + rnd_reward * 0.2
            gen_score = tf.reshape(generateScore_split, [self.batch_size, self.sample_buffer_size,
                                                         self.slate_num, -1])
            gen_score_mean = tf.reduce_mean(gen_score, axis=1, keep_dims=True)
            generateScore_split = (gen_score - gen_score_mean) * 10
            generateScore_split = tf.reshape(generateScore_split,
                                             [self.batch_size * self.sample_buffer_size * self.slate_num, -1])
        elif self.reward_type == "pbc_diversity":
            import numpy as np
            brand_id = tf.as_string(self.feature_dict['raw_brand_id_encode'])
            cate_id = tf.as_string(self.feature_dict['raw_cate_id_encode'])
            price_level = tf.as_string(self.feature_dict['raw_price_level'])
            bcp = brand_id + '-' + cate_id + '-' + price_level
            bcp = tf.gather_nd(bcp, self._criticfeatureIndex)
            
            def calc_diversity(x):
                x = x.tolist()
                diversity = []
                for i in range(len(x)):
                    cnt = len(set(x[i]))
                    diversity.append(float(cnt))
                diversity = np.array(diversity, dtype=np.float32)
                return diversity
    
            diver_reward = tf.py_func(calc_diversity, [bcp], [tf.float32])  # [B, 1]
            diver_reward = tf.reshape(diver_reward, [-1, 1])
            tf.summary.scalar("diversity_reward", tf.reduce_mean(diver_reward))
            self.pbc_diversity_cnt = tf.concat([self.pbc_diversity_cnt, diver_reward], axis=1)  # [B, K]
            last_is_new = tf.reshape(self.pbc_diversity_cnt[:, 9] - self.pbc_diversity_cnt[:, 8], [-1, 1])
            bcp_is_new = tf.concat([self.bcp_is_new, last_is_new], axis=1)
            # diversity reward
            # diver_base = tf.constant([[1., 1, 1, 2, 2, 2, 3, 3, 3, 3]])
            # diver_reward = tf.where(self.pbc_diversity_cnt < diver_base,
            #                         tf.ones_like(self.pbc_diversity_cnt, dtype=tf.float32),
            #                         tf.zeros_like(self.pbc_diversity_cnt, dtype=tf.float32))
            is_new_reward = tf.concat([tf.zeros_like(bcp_is_new[:, 0:1]), bcp_is_new[:, 1:]], axis=1) * 0.1
            is_new_reward = tf.where(self.pbc_diversity_cnt <= 3, is_new_reward, tf.zeros_like(is_new_reward))
            old_reward = (1 - bcp_is_new) * -0.02
            old_reward = tf.where(self.pbc_diversity_cnt <= 3, old_reward, tf.zeros_like(old_reward))
            neighbor_same = tf.concat([self.neighbor_same, tf.zeros(tf.shape(self.neighbor_same[:, -1:]))], axis=1)
            neighbor_same_reward = neighbor_same * -0.1
            
            # diversity_weight = self.train_conf['diversity_weight']
            generateScore_split = generateScore_split + is_new_reward + old_reward + neighbor_same_reward
        
        # subtract mean
        gen_score = tf.reshape(generateScore_split, [self.batch_size, self.sample_buffer_size,
                                                     self.slate_num, -1])
        gen_score_mean = tf.reduce_mean(gen_score, axis=1, keep_dims=True)
        generateScore_split = (gen_score - gen_score_mean) * 10
        generateScore_split = tf.reshape(generateScore_split,
                                         [self.batch_size * self.sample_buffer_size * self.slate_num, -1])
        
        # long reward for ppo
        if self.use_ppo == "true":
            a = tf.constant([0.01 * 0.95 * 0.95 * 0.95 * 0.95 * 0.95 * 0.95 * 0.95 * 0.95 * 0.95,
                             0.01 * 0.95 * 0.95 * 0.95 * 0.95 * 0.95 * 0.95 * 0.95 * 0.95,
                             0.01 * 0.95 * 0.95 * 0.95 * 0.95 * 0.95 * 0.95 * 0.95,
                             0.01 * 0.95 * 0.95 * 0.95 * 0.95 * 0.95 * 0.95, 0.01 * 0.95 * 0.95 * 0.95 * 0.95 * 0.95,
                             0.01 * 0.95 * 0.95 * 0.95 * 0.95, 0.01 * 0.95 * 0.95 * 0.95, 0.01 * 0.95 * 0.95,
                             0.01 * 0.95,
                             0.01])
            a = tf.to_float(a)
            self.generateScore_split_long = tf.tile(a, [self.batch_size * self.sample_buffer_size * self.slate_num])
            self.generateScore_split_long = tf.reshape(self.generateScore_split_long,
                                                       [self.batch_size * self.sample_buffer_size * self.slate_num, -1])
            self.long_reward = tf.reduce_sum(generateScore_split, axis=1)
            self.long_reward = tf.reshape(self.long_reward, [-1, 1])
            self.generateScore_split_long = self.generateScore_split_long * tf.tile(self.long_reward, [1, 10])
            self.generateScore_diff = generateScore_split - tf.slice(self._criticScore,
                                                                     [
                                                                         self.batch_size * self.sample_buffer_size * self.slate_num,
                                                                         0],
                                                                     [-1, -1])  # [B, K]
            generateScore_split = generateScore_split + self.generateScore_split_long
            # add exploration loss
            self.exploration_ratio = 1.0 - tf.div(tf.to_float(self.global_step), 2000000.0)
            generateScore_split = tf.add(generateScore_split, self.price_diversity_state)
            generateScore_split = tf.add(generateScore_split, self.embed_diversity_state)
        
        # 3. calculate loss of topK critic score
        self.critic_slate_score = tf.reduce_mean(generateScore_total)
        tf.summary.scalar("critic_score", self.critic_slate_score)
        
        # 4. get DQN for every score during generate
        self._targetQ = generateScore_total  # [B, 1] as reward.
        self._targetQ = tf.stop_gradient(self._targetQ)
        
        # REINFORCE loss (like AC) C as reward
        # generateScore_split = tf.clip_by_value(generateScore_split, 1e-10, 1.0)
        if self.use_ppo:
            # ppo loss
            ratio = self.finalScoreState / tf.stop_gradient(self.finalScoreState)
            surr_1 = tf.multiply(generateScore_split, ratio)
            surr_2 = tf.multiply(generateScore_split, tf.clip_by_value(ratio, 0.8, 1.2))
            self._actorLoss = tf.reduce_sum(tf.minimum(surr_1, surr_2), axis=1)
        else:
            self._actorLoss = tf.reduce_sum(tf.multiply(generateScore_split,
                                                        tf.log(tf.clip_by_value(self.finalScoreState, 1e-10, 1.0))),
                                            axis=1)  # [B, 1]
        
        self.generateScore_split = generateScore_split
        self._actorLoss = -1.0 * tf.reduce_mean(self._actorLoss)
        self._reward = tf.reduce_mean(self._targetQ)
        tf.summary.scalar("reward", self._reward)
        tf.summary.scalar("actor_loss", self._actorLoss)
    
    def load_critic_variables(self):
        variables = tf.trainable_variables()
        self.critic_variables = []
        for v in variables:
            print("variable: %s" % v.name)
            if v.name.find("critic_network") != -1:
                self.critic_variables.append(v)
        print("critic network variable number: %d" % len(self.critic_variables))
        self.critic_saver = tf.train.Saver(self.critic_variables)
        return self.critic_saver
    
    def backward(self, global_step):
        dnn_optimizer = None
        self.flags.train = self.model_conf['train']
        learning_rate = self.flags.train['learning_rate']
        # 1. define optimizer
        print("dnn_optimizer=%s, dnn_learning_rate=%f" % (self.flags.train['optimizer'], learning_rate))
        if self.flags.train['optimizer'] == "adam":
            dnn_optimizer = tf.train.AdamOptimizer(learning_rate)
        elif self.flags.train['optimizer'] == "adagrad":
            dnn_optimizer = tf.train.AdagradOptimizer(learning_rate)
        elif self.flags.train['optimizer'] == "momentum":
            dnn_optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=self.flags.train['dnn_momentum'],
                                                       use_nesterov=True)
        elif self.flags.train['optimizer'] == "rmsprop":
            dnn_optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=self.flags.train['dnn_momentum'])
        
        # 2. get moving average loss op
        if self.generate_loss_type == 'actor_loss':
            print('***use actor loss***')
            self.loss = self._actorLoss
        elif self.generate_loss_type == 'entropy':
            print('***use entropy loss***')
            self.loss = self._actorLoss - 0.01 * self.entropy
        
        ema = tf.train.ExponentialMovingAverage(decay=0.999, zero_debias=True)
        self.loss_ema = ema.apply([self.loss, self.critic_slate_score])
        avg_loss = ema.average(self.loss)
        tf.summary.scalar("avg_loss", avg_loss)
        criticScore = ema.average(self.critic_slate_score)
        tf.summary.scalar("avg_critic_score_reward", criticScore)
        
        # 3. get train variables
        dnn_variables = tf.trainable_variables()
        train_variables = []
        gradient_clip_norm = self.flags.train['gradient_clip_norm'] if 'gradient_clip_norm' in self.flags.train else 0.0
        print("dnn variables num: %d, gradient_clip_norm: %f" % (len(dnn_variables), gradient_clip_norm))
        for var in dnn_variables:
            print("dnn variable: %s" % var.name)
            if var.name.startswith(self.variable_scope):
                train_variables.append(var)
        print("train variables num: %d, gradient_clip_norm: %f" % (len(train_variables), gradient_clip_norm))
        
        # 4. get train op
        input_bn = self.flags.train['input_bn'] if 'input_bn' in self.flags.train else False
        if input_bn:
            bn_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        print("train with inputBn: %d" % input_bn)
        
        if gradient_clip_norm < 1e-5:
            if input_bn:
                with tf.control_dependencies(bn_update_ops):
                    self.train_op = tf.contrib.layers.optimize_loss(
                        loss=self.loss,
                        global_step=global_step,
                        learning_rate=None,  # using optimizer learning rate
                        optimizer=dnn_optimizer,
                        gradient_multipliers=None,
                        clip_gradients=None,
                        variables=train_variables)
            else:
                self.train_op = tf.contrib.layers.optimize_loss(
                    loss=self.loss,
                    global_step=global_step,
                    learning_rate=None,  # using optimizer learning rate
                    optimizer=dnn_optimizer,
                    gradient_multipliers=None,
                    clip_gradients=None,
                    variables=train_variables)
        else:
            # 5.2 has gradient clip norm
            print('using gradient_clip_norm: %f' % gradient_clip_norm)
            gradients = dnn_optimizer.compute_gradients(self.loss, train_variables)
            gradients_list = []
            var_list = []
            gradient_clip_type = self.flags.train[
                'gradient_clip_type'] if 'gradient_clip_type' in self.flags.train else None
            if gradient_clip_type == "norm":
                for g, v in gradients:
                    if g is not None:
                        gradients_list.append(g)
                        var_list.append(v)
                        print('using gradient clip by norm on variable: %s, norm: %f' % (v.name, gradient_clip_norm))
                
                clipGradientsList, _norm = tf.clip_by_global_norm(gradients_list, gradient_clip_norm)
                clipGradients = zip(clipGradientsList, var_list)
                tf.summary.scalar("gradient_norm", _norm)
            else:
                clipValueGradientsList = []
                for g, v in gradients:
                    if g is not None:
                        gradients_list.append(g)
                        var_list.append(v)
                        clipValueGradientsList.append(tf.clip_by_value(g, -gradient_clip_norm, gradient_clip_norm))
                        print('using gradient clip by norm on variable: %s, norm: %f' % (v.name, gradient_clip_norm))
                
                clipGradients = zip(clipValueGradientsList, var_list)
            # gradient clip done. just apply
            if input_bn:
                with tf.control_dependencies(bn_update_ops):
                    self.train_op = dnn_optimizer.apply_gradients(clipGradients, global_step=global_step)
            else:
                self.train_op = dnn_optimizer.apply_gradients(clipGradients, global_step=global_step)
    
    def get_outputs(self, data):
        FLAGS = self.flags
        # feature preprocess
        with tf.name_scope("featureparser_feature_input"):
            fg_configs = get_json_conf_from_file(FLAGS.feature_conf['input'])
            featureparser_features = featureparser_fg.parse_genreated_fg(fg_configs, data['features'])
            featureparser_features = format_feature_offline(featureparser_features, FLAGS.feature_conf['input'],
                                                  item_num=self.candidate_size)
        
        # only offline item_num is candidate_size
        self.item_num = self.candidate_size
        # build model
        net0_dict = self.get_net0_dict(featureparser_features)
        self.build_network(net0_dict)
        metrics = self.calc_metrics(self.get_var_dict(), featureparser_features)
        self.load_critic_variables()
        return metrics
    
    def build(self, data, global_step):
        FLAGS = self.flags
        # feature preprocess
        with tf.name_scope("featureparser_feature_input"):
            fg_configs = get_json_conf_from_file(FLAGS.feature_conf['input'])
            featureparser_features = featureparser_fg.parse_genreated_fg(fg_configs, data['features'])
            featureparser_features = format_feature_offline(featureparser_features, FLAGS.feature_conf['input'],
                                                  item_num=self.candidate_size)
            if self.reward_type == "add_real_reward":
                # label
                label_configs = get_json_conf_from_file(FLAGS.feature_conf['label'])
                label_dict = featureparser_fg.parse_genreated_fg(label_configs, data['label'])
                self.label_dict = format_feature_offline(label_dict, FLAGS.feature_conf['label'])
        
        # only offline item_num is candidate_size
        self.item_num = self.candidate_size
        self.global_step = global_step
        # build model
        net0_dict = self.get_net0_dict(featureparser_features)
        self.build_network(net0_dict)
        self.build_loss(featureparser_features)
        self.backward(global_step)
        self.load_critic_variables()
