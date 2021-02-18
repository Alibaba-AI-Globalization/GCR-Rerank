# coding=utf-8
"""
@author xxx
"""
from model_base import ModelBase
from transforms.preprocessing import Preprocess
from transforms.to_net0 import to_net0_dict
import json
import tensorflow as tf
from layers.layer_lib import build_dnn_net
from layers.layer_lib import build_logits
from transforms.data_format import format_feature_offline
from util.rtm import MetricsRT
from util import tf_summary
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope as vs
from transforms.data_format import get_json_conf_from_file
import featureparser_fg


class Model(ModelBase):
    def __init__(self, FLAGS, item_num=10, list_size=10, variable_scope="critic_network"):
        
        super(Model, self).__init__(name='Critic Model')
        self.flag = FLAGS
        self.preprocess_conf = FLAGS.feature_conf['critic_fpre']
        self.model_conf = None
        self.feature_conf = None
        self.label_conf = None
        self.feature_dict = None
        self.item_num = item_num    # 输入的 item 数量，可能小于 slate size
        self.list_size = list_size  # slate size 固定值
        self.batch_size = None
        self.variable_scope = variable_scope

        self.model_conf = FLAGS.critic_model
        with open(FLAGS.feature_conf['critic_features'], 'r') as fp:
            self.feature_conf = json.load(fp)
        with open(FLAGS.feature_conf['label'], 'r') as fp:
            self.label_conf = json.load(fp)
        
        self.logits = None
        self.predict_score = None
        self.clk_label = None
        self.atc_label = None
        self.pay_label = None
        self.label = None
        self.loss = None
        self.loss_ema = None
        self.train_op = None
        self.mrt = MetricsRT(FLAGS)
    
    def extend_pair_feature(self, item_features):
        batch_size = self.batch_size
        vec_dim = item_features.shape[1].value
        list_size = self.item_num
        # relative pos
        pos = tf.reshape(tf.tile(tf.range(1, list_size + 1, 1), [batch_size]), [batch_size, -1])
        pos1 = tf.reshape(tf.tile(pos, [1, list_size]), [-1, 1])
        pos2 = tf.reshape(tf.tile(tf.reshape(pos, [-1, 1]), [1, list_size]), [-1, 1])
        relative_pos = tf.to_float(tf.subtract(pos1, pos2)) * 0.1
        
        # i1,i1,i1;i2,i2,i2
        tile1 = tf.reshape(tf.tile(item_features, [1, list_size]), [-1, vec_dim])
        # i1,i2,i3;i1,i2,i3
        tile2 = tf.reshape(tf.tile(tf.reshape(item_features, [-1, list_size * vec_dim]), [1, list_size]), [-1, vec_dim])
        
        pair_features = tf.concat([tile1, tile2, relative_pos], 1)
        return pair_features
    
    def pair_net(self, pair_input):
        # pair net
        net_arch = self.model_conf['architecture']
        pair_net = build_dnn_net(pair_input, name='pair_net', hidden_units=net_arch['pair_net_units'],
                                 activation=net_arch['pair_net_activation'])
        pair_logits = build_logits(pair_net, name='pair_net')
        # pair net reduce sum
        pairnet_out_dim = net_arch['pair_net_units'][-1]
        pair_net_dim = tf.reshape(pair_net, [-1, self.item_num, self.item_num, pairnet_out_dim])
        pair_logit_dim = tf.nn.softmax(tf.reshape(pair_logits, [-1, self.item_num, self.item_num, 1]), dim=2)
        merge_dim = tf.multiply(pair_net_dim, pair_logit_dim)
        # pair net sum by one item (i1, i2; i1, i3; i1, i4;)
        sum_pair_net = tf.reshape(tf.reduce_sum(merge_dim, axis=2), [-1, pairnet_out_dim])
        return sum_pair_net
    
    def attention_net(self, net_input, name):
        # net_input [B, item_num, seq, N]
        net_arch = self.model_conf['architecture']
        atte_net = build_dnn_net(net_input, name=name, hidden_units=net_arch['attention_net_units'],
                                 activation=net_arch['attention_net_activation'])
        out_dim = net_input.shape[-1]
        atte_logits = build_logits(atte_net, name=name)  # [B, item_num, seq, 1]
        weight = tf.nn.softmax(atte_logits, dim=2)
        merge_net = tf.multiply(net_input, weight)
        sum_su_net = tf.reshape(tf.reduce_sum(merge_net, axis=2), [-1, out_dim])
        return sum_su_net
    
    def bi_gru_net(self, gru_input):
        print('gru input shape() = %s' % gru_input.get_shape())
        forward_gru_dimension = 256
        backward_gru_dimension = 256
        feature_num = gru_input.get_shape()[1]
        gru_input = tf.reshape(gru_input, [-1, self.item_num, feature_num])
        # pad at axis = 1 and slice at least 10.
        padInputTensor = tf.fill([tf.shape(gru_input)[0], 10, feature_num], 0.0)
        gru_input = tf.concat([gru_input, padInputTensor], axis=1)
        gru_input = tf.slice(gru_input, [0, 0, 0], [-1, 10, feature_num])
        # pad done, unstack and gru
        inputList = tf.unstack(gru_input, axis=1)
        
        fwCell = tf.contrib.rnn.GRUCell(num_units=forward_gru_dimension)
        bwCell = tf.contrib.rnn.GRUCell(num_units=backward_gru_dimension)
        hiddenList, fwState, bwState = tf.contrib.rnn.static_bidirectional_rnn(cell_fw=fwCell, cell_bw=bwCell,
                                                                               inputs=inputList, dtype=tf.float32)
        # get hiddenList for every item
        hidden_size = forward_gru_dimension + backward_gru_dimension
        zeroPadding = tf.fill(tf.shape(tf.reshape(hiddenList[0], [-1, 1, hidden_size])), 0.0)
        forwardHiddenList = [zeroPadding]
        backwardHiddenList = []
        for idx in range(0, len(hiddenList) - 1):
            forwardHiddenList.append(tf.reshape(hiddenList[idx], [-1, 1, hidden_size]))
        for idx in range(1, len(hiddenList)):
            backwardHiddenList.append(tf.reshape(hiddenList[idx], [-1, 1, hidden_size]))
        backwardHiddenList.append(zeroPadding)
        forwardNet = tf.concat(forwardHiddenList, axis=1)
        forwardNet = tf.slice(forwardNet, [0, 0, 0],
                              [-1, self.item_num, forward_gru_dimension])
        forwardNet = tf.reshape(forwardNet, [-1, forward_gru_dimension])
        backwardNet = tf.concat(backwardHiddenList, axis=1)
        backwardNet = tf.slice(backwardNet, [0, 0, forward_gru_dimension],
                               [-1, self.item_num, backward_gru_dimension])
        backwardNet = tf.reshape(backwardNet, [-1, backward_gru_dimension])
        print("forwardNet %s, backwardNet shape: %s" % (forwardNet.get_shape(), backwardNet.get_shape()))
        tf_summary.add_hidden_layer_summary(forwardNet, "gru_forward")
        tf_summary.add_hidden_layer_summary(backwardNet, "gru_backward")
        tf_summary.add_hidden_layer_summary(forwardNet, "gru_forward")
        tf_summary.add_hidden_layer_summary(backwardNet, "gru_backward")
        return forwardNet, backwardNet
    
    def position_embedding(self):
        bucket_size = self.list_size
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
    
    def add_trick_features(self):
        # add item_pos
        item_pos = self.position_embedding()
        if 'search_pos' in self.feature_dict:
            search_pos = self.feature_dict['search_pos']
            search_pos = tf.reshape(search_pos, [-1, search_pos.shape[-1]])
            self.feature_dict['item_pos'] = item_pos + search_pos
        else:
            self.feature_dict['item_pos'] = item_pos
        
        # add price_gap
        if "real_price" in self.feature_dict and "avg_price_7d" in self.feature_dict:
            print "add price_gap"
            real_price = self.feature_dict["real_price"]
            avg_price_7d = self.feature_dict["avg_price_7d"]
            avg_price_7d = tf.where(tf.equal(avg_price_7d, 0), real_price, avg_price_7d)
            avg_price_7d = tf.where(tf.equal(avg_price_7d, 0), tf.ones_like(avg_price_7d, dtype=tf.float32),
                                    avg_price_7d)
            price_gap = tf.clip_by_value(real_price / avg_price_7d - 1, -1, 1)
            self.feature_dict["price_gap"] = tf.reshape(price_gap, [-1, 1])
    
    def get_net0_dict(self, featureparser_feature_dict):
        with tf.name_scope("critic_feature_pipe"):
            preprocess = Preprocess(self.preprocess_conf)
            self.feature_dict = preprocess.process(featureparser_feature_dict, list_size=self.item_num,
                                                   variable_scope=self.variable_scope)
            self.batch_size = preprocess.batch_size
            self.add_trick_features()
            net0_dict = to_net0_dict(self.feature_dict, conf=self.feature_conf, list_size=self.list_size)
            # pair_features
            item_dense = net0_dict['item_dense']
            pair_features = self.extend_pair_feature(item_dense)
            net0_dict['pair_features'] = pair_features

            user_pv_seq_embed = net0_dict['user_pv_seq']   # [B, seq, N]
            user_click_seq_embed = net0_dict['user_click_seq']
            user_atc_seq_embed = net0_dict['user_atc_seq']
            user_pay_seq_embed = net0_dict['user_pay_seq']
            # [B, item, seq, N]
            user_pv_seq_embed = tf.tile(tf.expand_dims(user_pv_seq_embed, axis=1), [1, self.item_num, 1, 1])
            user_click_seq_embed = tf.tile(tf.expand_dims(user_click_seq_embed, axis=1), [1, self.item_num, 1, 1])
            user_atc_seq_embed = tf.tile(tf.expand_dims(user_atc_seq_embed, axis=1), [1, self.item_num, 1, 1])
            user_pay_seq_embed = tf.tile(tf.expand_dims(user_pay_seq_embed, axis=1), [1, self.item_num, 1, 1])
            
            item_id_features = net0_dict['item_id_features']  # [B*item, N]
            n_dim = item_id_features.shape[1]
            item_id_features = tf.reshape(item_id_features, [-1, self.item_num, 1, n_dim])
            # [B, item, seq, N]
            seq = tf.shape(user_pv_seq_embed)[2]
            item_id_features = tf.tile(item_id_features, [1, 1, seq, 1])
            
            su_pv_input = tf.concat([item_id_features, user_pv_seq_embed], axis=3)
            su_click_input = tf.concat([item_id_features, user_click_seq_embed], axis=3)
            su_atc_input = tf.concat([item_id_features, user_atc_seq_embed], axis=3)
            su_pay_input = tf.concat([item_id_features, user_pay_seq_embed], axis=3)

            with tf.variable_scope(self.variable_scope, reuse=tf.AUTO_REUSE):
                su_pv = self.attention_net(su_pv_input, name='su_pv_attention')  # [B*item, N]
                su_click = self.attention_net(su_click_input, name='su_click_attention')
                su_atc = self.attention_net(su_atc_input, name='su_atc_attention')
                su_pay = self.attention_net(su_pay_input, name='su_pay_attention')

                net_arch = self.model_conf['architecture']
                if 'model_type' in net_arch and net_arch['model_type'] == 'cvr_loss':
                    self.cvr_logits = build_logits(su_click, name='su_cvr_net')
                    self.cvr_predict_score = tf.sigmoid(self.cvr_logits)
                    

            user_embed = tf.concat([su_pv, su_click, su_atc, su_pay], axis=1)

            user_seq_dense = net0_dict['user_seq_dense']  # [B, seq]
            print "  attention net, user_embed, shape: %s", user_embed.get_shape()
            net0_dict['su_rt'] = user_embed
            net0_dict['item_dense'] = tf.concat([item_dense, user_embed], axis=1)
            tf_summary.add_net0_summary(net0_dict)
        
        return net0_dict
    
    def build_network(self, net0_dict):
        dnn_input = net0_dict['item_dense']
        context_features = net0_dict['context_features']
        pair_features = net0_dict['pair_features']
        gru_input = tf.concat([dnn_input, context_features], axis=1)
        
        net_arch = self.model_conf['architecture']
        with tf.variable_scope(self.variable_scope, reuse=tf.AUTO_REUSE):
            # dnn net, context net
            dnn_net = build_dnn_net(dnn_input, name='dnn_net', hidden_units=net_arch['dnn_units'],
                                    activation=net_arch['dnn_activation'])
            context_net = build_dnn_net(context_features, name='context_net', hidden_units=net_arch['dnn_units'],
                                        activation=net_arch['dnn_activation'])
            # pair net, gru net
            pair_net = self.pair_net(pair_features)
            forward_net, backward_net = self.bi_gru_net(gru_input)
            # predict net
            merge = tf.concat([dnn_net, context_net, pair_net, forward_net, backward_net], axis=1)
            if 'add_diversity' in net_arch and net_arch['add_diversity'] is True:
                print("add diversity features")
                shop_ratio = tf.reduce_sum(self.feature_dict['shop_id_encode_same'], axis=1, keep_dims=True) / 10
                bu_ratio = tf.reduce_sum(self.feature_dict['bu_id_encode_same'], axis=1, keep_dims=True) / 10
                price_ratio = tf.reduce_sum(self.feature_dict['price_same'], axis=1, keep_dims=True) / 10
                brand_ratio = tf.reduce_sum(self.feature_dict['brand_id_encode_same'], axis=1, keep_dims=True) / 10
                cate_ratio = tf.reduce_sum(self.feature_dict['cate_id_encode_same'], axis=1, keep_dims=True) / 10
                seller_ratio = tf.reduce_sum(self.feature_dict['seller_id_encode_same'], axis=1, keep_dims=True) / 10
                merge = tf.concat([merge, shop_ratio, bu_ratio, price_ratio, brand_ratio, cate_ratio, seller_ratio], axis=1)
            if 'query' in net0_dict:
                query = tf.reduce_mean(net0_dict['query'], axis=1, keep_dims=True)
                query = tf.tile(query, [1, self.item_num, 1])
                query = tf.reshape(query, [-1, query.get_shape()[2]])
                if 'usr_curiosity_vector' in net0_dict:
                    print('add usr_curiosity_vector:', net0_dict['usr_curiosity_vector'].shape[1])
                    usr_curiosity_vector = tf.expand_dims(net0_dict['usr_curiosity_vector'], axis=1)
                    usr_curiosity_vector = tf.tile(usr_curiosity_vector, [1, self.item_num, 1])
                    usr_curiosity_vector = tf.reshape(usr_curiosity_vector, [-1, usr_curiosity_vector.get_shape()[2]])
                    query = tf.concat([query, usr_curiosity_vector], axis=1)
                query_vec = build_dnn_net(query, name='bilinear', hidden_units=[256], activation=[None])
                merge_vec = build_dnn_net(merge, name='pred_net_0', hidden_units=[256], activation=[None])
                merge = tf.multiply(query_vec, merge_vec)
            predict_net = build_dnn_net(merge, name='pred_net', hidden_units=net_arch['predict_net_units'],
                                        activation=net_arch['predict_net_activation'])
            
            if 'model_type' in net_arch and net_arch['model_type'] == 'ESMM':
                cvr_net = build_dnn_net(merge, name='cvr_net', hidden_units=net_arch['predict_net_units'],
                                            activation=net_arch['predict_net_activation'])
                self.cvr_logits = build_logits(cvr_net, name='cvr_net')
                self.cvr_predict_score = tf.sigmoid(self.cvr_logits)
                # ctr
                self.ctr_logits = build_logits(predict_net, name='ctr_net')
                self.ctr_predict_score = tf.sigmoid(self.ctr_logits)
                # pvpay
                self.predict_score = self.ctr_predict_score * self.cvr_predict_score
                self.logits = self.predict_score
                return self.predict_score
            elif 'model_type' in net_arch and net_arch['model_type'] == 'gmsl':
                cvr_net = build_dnn_net(merge, name='cvr_net', hidden_units=net_arch['predict_net_units'],
                                            activation=net_arch['predict_net_activation'])
                self.cvr_logits = build_logits(cvr_net, name='cvr_net')
                self.cvr_predict_score = tf.sigmoid(self.cvr_logits)
                # ctr
                self.ctr_logits = build_logits(predict_net, name='ctr_net')
                self.ctr_predict_score = tf.sigmoid(self.ctr_logits)
                # build shared GRU
                cell = tf.nn.rnn_cell.GRUCell(num_units=64)
                cvr_net_gru, cvr_state_gru = cell(cvr_net, predict_net)
                # pvpay
                self.logits = build_logits(cvr_net_gru, name='pvp_net')
                self.predict_score = tf.sigmoid(self.logits)
                return self.predict_score
            elif 'model_type' in net_arch and net_arch['model_type'] == 'bias_net':
                query_bias_net = build_dnn_net(query, name='query_bias_net', hidden_units=[256, 128, 64],
                                               activation=['relu', 'relu', 'relu'])
                self.query_logits = build_logits(query_bias_net, name='query_bias_net')
                self.logits = build_logits(predict_net, name='pred_net') + self.query_logits
                self.predict_score = tf.sigmoid(self.logits)
                return self.predict_score
            elif 'model_type' in net_arch and net_arch['model_type'] == 'bias_net2':
                bias_net_input = net0_dict['bias_net_input']
                bias_net = build_dnn_net(bias_net_input, name='bias_net', hidden_units=[32, 32],
                                         activation=['relu', 'relu'])
                self.bias_logits = build_logits(bias_net, name='bias_net')
                self.logits = build_logits(predict_net, name='pred_net') + self.bias_logits
                self.predict_score = tf.sigmoid(self.logits)
                return self.predict_score
            
            self.logits = build_logits(predict_net, name='pred_net')
            self.predict_score = tf.sigmoid(self.logits)
        
        return self.predict_score
    
    def get_label(self, label_dict):
        label_dict = format_feature_offline(label_dict, self.flag.feature_conf['label'])
        self.flag.train = self.model_conf['train']
        label_type = self.flag.train['label_type']
        if label_type == 'merge':
            self.clk_label = clk_label = tf.reshape(label_dict['click'], [-1, 1])
            self.atc_label = atc_label = tf.reshape(label_dict['atc'], [-1, 1])
            self.pay_label = pay_label = tf.reshape(label_dict['pay'], [-1, 1])
            self.label = label = tf.clip_by_value(atc_label + pay_label, 0, 1)
            if 'sample_weights' in self.flag.train:
                pv_w, pst_w, atc_w, pay_w = [float(i) for i in self.flag.train['sample_weights'].split(",")]
            else:
                pv_w, pst_w, atc_w, pay_w = 0.05, 1.0, 3.0, 46.0
            print("critic sample weights = %f, %f, %f, %f" % (pv_w, pst_w, atc_w, pay_w))
            pv_weight = tf.cast(tf.less(clk_label, 1), tf.float32) * pv_w
            pst_weight = tf.cast(tf.greater_equal(clk_label, 1), tf.float32) * pst_w
            atc_weight = tf.cast(tf.greater_equal(atc_label, 1), tf.float32) * atc_w
            pay_weight = tf.cast(tf.greater_equal(pay_label, 1), tf.float32) * pay_w
            label_weight = pv_weight + pst_weight + atc_weight + pay_weight
        elif label_type == 'gmv':
            self.clk_label = clk_label = tf.reshape(label_dict['click'], [-1, 1])
            self.atc_label = atc_label = tf.reshape(label_dict['atc'], [-1, 1])
            self.pay_label = pay_label = tf.reshape(label_dict['pay'], [-1, 1])
            self.label = label = tf.clip_by_value(atc_label + pay_label, 0, 1)
            gmv = tf.reshape(label_dict['gmv'], [-1, 1])
            pv_w, pst_w, atc_w, pay_w = 0.05, 0.5, 2.0, gmv
            print("gmv critic weights = %f, %f, %f, gmv" % (pv_w, pst_w, atc_w))
            
            pv_weight = tf.cast(tf.less(clk_label, 1), tf.float32) * pv_w
            pst_weight = tf.cast(tf.greater_equal(clk_label, 1), tf.float32) * pst_w
            atc_weight = tf.cast(tf.greater_equal(atc_label, 1), tf.float32) * atc_w
            pay_weight = tf.cast(tf.greater_equal(pay_label, 1), tf.float32) * pay_w
            label_weight = pv_weight + pst_weight + atc_weight + pay_weight
        else:
            self.clk_label = clk_label = tf.reshape(label_dict['click'], [-1, 1])
            self.atc_label = atc_label = tf.reshape(label_dict['atc'], [-1, 1])
            self.pay_label = pay_label = tf.reshape(label_dict['pay'], [-1, 1])
            self.label = label = tf.reshape(label_dict['pay'], [-1, 1])
            label_weight = None
        tf.summary.scalar("label/fraction_of_zero_values", tf.nn.zero_fraction(label))
        return label, label_weight
    
    def build_loss(self, label_dict):
        self.flag.train = self.model_conf['train']
        self.flag.architecture = self.model_conf['architecture']
        # 1. get label
        with tf.name_scope("critic_label_input"):
            label, label_weight = self.get_label(label_dict)
        
        # 2. get loss op
        num_class = self.flag.architecture['num_class'] if 'num_class' in self.flag.architecture else 2
        loss_type = self.flag.train['loss_type'] if 'loss_type' in self.flag.architecture else 2
        net_arch = self.flag.architecture
        if 'model_type' in net_arch and net_arch['model_type'] == 'ESMM':
            label_weight = None
            ctr_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.clk_label, tf.float32),
                                                                        logits=self.ctr_logits,
                                                                        name='ctr_logits_cross_entropy')
            cvr_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.pay_label, tf.float32),
                                                                        logits=self.cvr_logits,
                                                                        name='cvr_logits_cross_entropy')
            cvr_cross_entropy = cvr_cross_entropy * self.clk_label
            loss_11 = -tf.log(tf.clip_by_value(self.ctr_predict_score * self.cvr_predict_score, 1e-10, 1))
            loss_10 = -tf.log(tf.clip_by_value(self.ctr_predict_score * (1 - self.cvr_predict_score), 1e-10, 1))
            loss_00 = -tf.log(1 - self.ctr_predict_score) - \
                      tf.log(tf.clip_by_value(1 - (self.ctr_predict_score*self.ctr_predict_score), 1e-10, 1))
            pvpay_loss = (loss_11 * self.clk_label * self.pay_label +
                          loss_10 * self.clk_label * (1 - self.pay_label) +
                          loss_00 * (1 - self.clk_label) * (1 - self.pay_label))
            # cross_entropy = 0.1 * pvpay_loss + 0.01 * ctr_cross_entropy + 2 * cvr_cross_entropy
            ctr_sigma = tf.Variable(1., name='ctr_sigma')
            cvr_sigma = tf.Variable(1., name='cvr_sigma')
            pvpay_sigma = tf.Variable(1., name='pvpay_sigma')
            cross_entropy = tf.exp(-1 * ctr_sigma) * ctr_cross_entropy + tf.exp(-1 * cvr_sigma) * cvr_cross_entropy + \
                            tf.exp(-1 * pvpay_sigma) * pvpay_loss + pvpay_sigma + ctr_sigma + cvr_sigma
            tf.summary.scalar("ctr_cross_entropy", tf.reduce_mean(ctr_cross_entropy))
            tf.summary.scalar("cvr_cross_entropy", tf.reduce_mean(cvr_cross_entropy))
            tf.summary.scalar("pvpay_loss", tf.reduce_mean(pvpay_loss))
            tf.summary.scalar("weight_loss/ctr_cross_entropy", tf.reduce_mean(tf.exp(-1 * ctr_sigma) * ctr_cross_entropy))
            tf.summary.scalar("weight_loss/cvr_cross_entropy", tf.reduce_mean(tf.exp(-1 * cvr_sigma) * cvr_cross_entropy))
            tf.summary.scalar("weight_loss/pvpay_loss", tf.reduce_mean(tf.exp(-1 * pvpay_sigma) * pvpay_loss))
            tf.summary.scalar("sigma/ctr_sigma", ctr_sigma)
            tf.summary.scalar("sigma/cvr_sigma", cvr_sigma)
            tf.summary.scalar("sigma/pvpay_sigma", pvpay_sigma)
        elif 'model_type' in net_arch and net_arch['model_type'] == 'gmsl':
            label_weight = None
            ctr_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.clk_label, tf.float32),
                                                                        logits=self.ctr_logits,
                                                                        name='ctr_logits_cross_entropy')
            cvr_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.pay_label, tf.float32),
                                                                        logits=self.cvr_logits,
                                                                        name='cvr_logits_cross_entropy')
            cvr_cross_entropy = cvr_cross_entropy * self.clk_label
            pvpay_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.pay_label, tf.float32),
                                                                        logits=self.logits,
                                                                        name='pvp_logits_cross_entropy')
            ctr_sigma = tf.Variable(1., name='ctr_sigma')
            cvr_sigma = tf.Variable(1., name='cvr_sigma')
            pvpay_sigma = tf.Variable(1., name='pvpay_sigma')
            cross_entropy = tf.exp(-1 * ctr_sigma) * ctr_cross_entropy + tf.exp(-1 * cvr_sigma) * cvr_cross_entropy + \
                            tf.exp(-1 * pvpay_sigma) * pvpay_loss + pvpay_sigma + ctr_sigma + cvr_sigma
            tf.summary.scalar("ctr_cross_entropy", tf.reduce_mean(ctr_cross_entropy))
            tf.summary.scalar("cvr_cross_entropy", tf.reduce_mean(cvr_cross_entropy))
            tf.summary.scalar("pvpay_loss", tf.reduce_mean(pvpay_loss))
            tf.summary.scalar("weight_loss/ctr_cross_entropy", tf.reduce_mean(tf.exp(-1 * ctr_sigma) * ctr_cross_entropy))
            tf.summary.scalar("weight_loss/cvr_cross_entropy", tf.reduce_mean(tf.exp(-1 * cvr_sigma) * cvr_cross_entropy))
            tf.summary.scalar("weight_loss/pvpay_loss", tf.reduce_mean(tf.exp(-1 * pvpay_sigma) * pvpay_loss))
            tf.summary.scalar("sigma/ctr_sigma", ctr_sigma)
            tf.summary.scalar("sigma/cvr_sigma", cvr_sigma)
            tf.summary.scalar("sigma/pvpay_sigma", pvpay_sigma)
        elif 'model_type' in net_arch and net_arch['model_type'] == 'cvr_loss':
            cvr_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.pay_label, tf.float32),
                                                                        logits=self.cvr_logits,
                                                                        name='cvr_logits_cross_entropy')
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(label, tf.float32),
                                                                    logits=self.logits,
                                                                    name='logits_cross_entropy')
            cross_entropy = cross_entropy + 0.1 * cvr_cross_entropy
        elif 'model_type' in net_arch and net_arch['model_type'] == 'bias_net2':
            bias_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.pay_label, tf.float32),
                                                                         logits=self.bias_logits,
                                                                         name='bias_logits_cross_entropy')
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(label, tf.float32),
                                                                    logits=self.logits,
                                                                    name='logits_cross_entropy')
            cross_entropy = cross_entropy + 0.1 * bias_cross_entropy
        elif num_class == 2:
            if loss_type == "focal_loss":
                print("use focal loss.")
                from layers.loss_layer import focal_loss
                cross_entropy = focal_loss(prediction=self.predict_score, target=tf.cast(label, tf.float32))
            else:
                cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(label, tf.float32),
                                                                        logits=self.logits,
                                                                        name='logits_cross_entropy')
        else:
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=self.logits)
        
        if label_weight is not None:
            cross_entropy = tf.losses.compute_weighted_loss(cross_entropy, label_weight,
                                                            reduction=tf.losses.Reduction.NONE)
        self.loss = tf.reduce_mean(cross_entropy, name='loss_reduce_cross_entropy')
        
        dnn_l2 = self.flag.train['dnn_l2'] if 'dnn_l2' in self.flag.train else 0.0
        if dnn_l2 > 1e-6:
            l2_loss = tf.losses.get_regularization_loss()
            tf.summary.scalar('l2_loss', l2_loss)
            self.loss += l2_loss
        tf.summary.scalar("loss", self.loss)
        return self.loss
    
    def backward(self, global_step=None):
        dnn_optimizer = None
        learning_rate = self.flag.train['learning_rate']
        # 1. define optimizer
        print("dnn_optimizer=%s, dnn_learning_rate=%f" % (self.flag.train['optimizer'], learning_rate))
        if self.flag.train['optimizer'] == "adam":
            dnn_optimizer = tf.train.AdamOptimizer(learning_rate)
        elif self.flag.train['optimizer'] == "adagrad":
            dnn_optimizer = tf.train.AdagradOptimizer(learning_rate)
        elif self.flag.train['optimizer'] == "momentum":
            dnn_optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=self.flag.train['dnn_momentum'],
                                                       use_nesterov=True)
        elif self.flag.train['optimizer'] == "rmsprop":
            dnn_optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=self.flag.train['dnn_momentum'])
        
        # 2. get moving average loss op
        ema = tf.train.ExponentialMovingAverage(decay=0.999, zero_debias=True)
        self.loss_ema = ema.apply([self.loss])
        avg_loss = ema.average(self.loss)
        tf.summary.scalar("avgLoss", avg_loss)
        
        # 3. get train variables
        dnn_variables = tf.trainable_variables()
        train_variables = []
        gradient_clip_norm = self.flag.train['gradient_clip_norm'] if 'gradient_clip_norm' in self.flag.train else 0.0
        print("dnn variables num: %d, gradient_clip_norm: %f" % (len(dnn_variables), gradient_clip_norm))
        for var in dnn_variables:
            print("dnn variable: %s" % var.name)
            train_variables.append(var)
        print("train variables num: %d, gradient_clip_norm: %f" % (len(train_variables), gradient_clip_norm))
        
        # 4. get train op
        input_bn = self.flag.train['input_bn'] if 'input_bn' in self.flag.train else False
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
            pass
    
    def auc_summary(self):
        self.mrt.add_auc(pred=self.predict_score, label=self.clk_label, mask=None, name='AUC_ctr')
        self.mrt.add_auc(pred=self.predict_score, label=self.pay_label, mask=self.clk_label, name='AUC_cvr')
        self.mrt.add_auc(pred=self.predict_score, label=self.pay_label, mask=None, name='AUC_pvp')
        self.mrt.add_auc(pred=self.predict_score, label=self.label, mask=None, name='AUC_atc')
    
    def get_outputs(self, data):
        FLAGS = self.flag
        outfield_json = {
            "features": [
                {
                    "feature_name": "item_id",
                    "feature_type": "id_feature",
                    "value_type": "String",
                    "expression": "item:item_id"
                },
                {
                    "feature_name": "featureparser_trace_critic_score",
                    "feature_type": "id_feature",
                    "value_type": "Float",
                    "expression": "item:featureparser_trace_critic_score"
                }
            ]
        }
        # feature preprocess
        with tf.name_scope("featureparser_feature_input"):
            fg_configs = get_json_conf_from_file(FLAGS.feature_conf['input'])
            featureparser_features = featureparser_fg.parse_genreated_fg(fg_configs, data['features'])
            featureparser_features = format_feature_offline(featureparser_features, FLAGS.feature_conf['input'])
            # label
            label_configs = get_json_conf_from_file(FLAGS.feature_conf['label'])
            label_dict = featureparser_fg.parse_genreated_fg(label_configs, data['label'])
            label_dict = format_feature_offline(label_dict, FLAGS.feature_conf['label'])
            # output field
            outfields = featureparser_fg.parse_genreated_fg(outfield_json, data['features'])
            outfields = format_feature_offline(outfields, outfield_json)
            list_size = 10
            search_id = tf.expand_dims(data['search_id'], 1)
            search_ids = tf.reshape(tf.tile(search_id, [1, list_size]), [-1, 1])
            rn = tf.expand_dims(data['rn'], 1)
            rns = tf.reshape(tf.tile(rn, [1, list_size]), [-1, 1])
            item_id = tf.reshape(outfields["item_id"], [-1, 1])
        
        net0_dict = self.get_net0_dict(featureparser_features)
        self.build_network(net0_dict)
        
        print("predict result table: {}".format(FLAGS.data['critic_result_table']))
        writer = tf.TableRecordWriter(FLAGS.data['critic_result_table'], slice_id=FLAGS.task_index)
        write_op = writer.write([0, 1, 2, 3, 4, 5, 6, 7],
                                [search_ids, rns, item_id, self.logits, self.predict_score, label_dict['pay'],
                                 outfields['featureparser_trace_critic_score'], label_dict['click']])
        close_writer_op = writer.close()
        return write_op, close_writer_op
    
    def build(self, data, global_step):
        FLAGS = self.flag
        # feature preprocess
        with tf.name_scope("featureparser_feature_input"):
            fg_configs = get_json_conf_from_file(FLAGS.feature_conf['input'])
            self.featureparser_parse_features = featureparser_fg.parse_genreated_fg(fg_configs, data['features'])
            featureparser_features = format_feature_offline(self.featureparser_parse_features, FLAGS.feature_conf['input'])
            # label
            label_configs = get_json_conf_from_file(FLAGS.feature_conf['label'])
            label_dict = featureparser_fg.parse_genreated_fg(label_configs, data['label'])
        
        # build model
        net0_dict = self.get_net0_dict(featureparser_features)
        self.build_network(net0_dict)
        self.build_loss(label_dict)
        self.backward(global_step)
        self.auc_summary()

    def get_featureparser_parse(self):
        return self.featureparser_parse_features

