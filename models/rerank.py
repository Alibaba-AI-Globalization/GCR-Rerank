"""
@author xxx
"""
import tensorflow as tf
from model_base import ModelBase
from datetime import datetime
from transforms.data_format import get_feature_conf_dict, get_json_conf_from_file
import importlib
import json
import traceback
from transforms.tensor_op import pad_sparse2dense


def time_log(func):
    def wrapper(*args, **kwargs):
        name = func.__name__
        print(str(datetime.now()) + " " + name + " start...")
        result = func(*args, **kwargs)
        print(str(datetime.now()) + " " + name + " finish...")
        return result
    
    return wrapper


class Model(ModelBase):
    def __init__(self, FLAGS):
        
        super(Model, self).__init__(name='Search Rerank Model')
        self.flag = FLAGS
        self.list_size = 10
        self.rerank_num = 50
        self.g_candidate_size = 50
        
        with open(FLAGS.feature_conf['input'], 'r') as fp:
            config = json.load(fp)
        self.feature_config = get_feature_conf_dict(config)
        
        self.rerank_type = FLAGS.rerank_type
        
        # sub model saver
        self.critic_variables = []
        self.cr_variables = []
        self.asp_variables = []
        self.diversity_variables = []
        self.actor_variables = []
        self.critic_saver = None
        self.generator_saver = None
        self.asp_saver = None
        self.diversity_saver = None
        self.actor_saver = None
        self.cr_generator_scope = "generator_network"
        self.diversity_generator_scope = "diversity_generator_net"
        self.asp_generator_scope = "asp_generator_net"
    
    def build_network(self, featureparser_features):
        # 1. format input features
        feature_dict, input_item_num = self.format_feature_online(featureparser_features, self.flag.feature_conf['input'])
        search_page = tf.reshape(feature_dict['search_pos'], [-1])[0]
        feature_dict = self.tile_search_pos(feature_dict, input_item_num)
        # 2. get topn to rerank
        self.feature_dict = tf.cond(input_item_num > self.rerank_num,
                                    lambda: self.get_topn_auction(feature_dict, self.rerank_num),
                                    lambda: feature_dict)
        # 3. get real rerank item number
        self.item_num = tf.where(input_item_num > self.rerank_num, self.rerank_num, input_item_num)
        self.list_size = tf.where(self.item_num > 10, 10, self.item_num)
        
        # 4. add generator model
        generator_lzd = importlib.import_module('models.%s' % self.flag.generator_model['name'])
        print("rerank_type: {}".format(self.rerank_type))
        if self.rerank_type in ("cr", "generator_only", "asp", "asp2", "gmv", "user_power_asp", "user_power_asp2"):
            self.generator = generator = generator_lzd.Model(self.flag, input_item_num=self.item_num,
                                                             candidate_size=self.g_candidate_size, batch_size=1)
            net0_dict = generator.get_net0_dict(self.feature_dict)
            gen_indices = generator.build_network(net0_dict)
            gen_indices = tf.cast(gen_indices, dtype=tf.int32)
            # self.generator_slate_score = tf.reduce_prod(generator.generator_score, axis=1)
            self.generator_score = generator.generator_score
        elif self.rerank_type in ("cr_asp", "cr_asp2", "cr_asp3"):
            # cr generator
            self.generator = cr_generator = generator_lzd.Model(self.flag, input_item_num=self.item_num,
                                                                candidate_size=self.g_candidate_size, batch_size=1,
                                                                variable_scope=self.cr_generator_scope)
            net0_dict = cr_generator.get_net0_dict(self.feature_dict)
            cr_generator.pbc_diversity = False
            cr_generator.isp_mode = False
            cr_generator.user_power_mode = False
            gen_indices = cr_generator.build_network(net0_dict)
            # asp generator
            self.asp_generator = asp_generator = generator_lzd.Model(
                self.flag, input_item_num=self.item_num, candidate_size=self.g_candidate_size, batch_size=1,
                variable_scope=self.asp_generator_scope)
            net0_dict = asp_generator.get_net0_dict(self.feature_dict)
            asp_generator.isp_mode = False
            asp_generator.user_power_mode = True
            asp_indices = asp_generator.build_network(net0_dict)
            gen_indices = tf.concat([gen_indices, asp_indices], axis=0)
            gen_indices = tf.cast(gen_indices, dtype=tf.int32)
            # self.generator_slate_score = tf.reduce_prod(cr_generator.generator_score, axis=1)
            self.generator_score = cr_generator.generator_score
        elif self.rerank_type == "page_cr_asp":
            # cr: 0, asp: 1; page <=2
            action = tf.where(tf.less_equal(search_page, 10), 1, 0)
            # cr generator
            generator = importlib.import_module('models.generator_lzd')
            self.generator = cr_generator = generator.Model(self.flag, input_item_num=self.item_num,
                                                                candidate_size=self.g_candidate_size, batch_size=1,
                                                                variable_scope=self.cr_generator_scope)
            net0_dict = cr_generator.get_net0_dict(self.feature_dict)
            gen_indices = cr_generator.build_network(net0_dict)
            # asp generator
            self.asp_generator = asp_generator = generator_lzd.Model(
                self.flag, input_item_num=self.item_num, candidate_size=self.g_candidate_size, batch_size=1,
                variable_scope=self.asp_generator_scope)
            net0_dict = asp_generator.get_net0_dict(self.feature_dict)
            asp_indices = asp_generator.build_network(net0_dict)
            # choose cr or asp model
            gen_indices = tf.where(tf.equal(action, 1), asp_indices, gen_indices)
            gen_indices = tf.cast(gen_indices, dtype=tf.int32)
            self.generator_score = tf.where(tf.equal(action, 1), asp_generator.generator_score,
                                            self.generator.generator_score)
        elif self.rerank_type in ("cr_asp_gmv", "cr_asp_gmv2", "cr_asp_gmv3"):
            # cr generator
            self.generator = cr_generator = generator_lzd.Model(self.flag, input_item_num=self.item_num,
                                                                candidate_size=self.g_candidate_size, batch_size=1,
                                                                variable_scope=self.cr_generator_scope)
            net0_dict = cr_generator.get_net0_dict(self.feature_dict)
            cr_generator.user_power_mode = False
            gen_indices = cr_generator.build_network(net0_dict)
            # asp generator
            self.asp_generator = asp_generator = generator_lzd.Model(
                self.flag, input_item_num=self.item_num, candidate_size=self.g_candidate_size, batch_size=1,
                variable_scope=self.asp_generator_scope)
            net0_dict = asp_generator.get_net0_dict(self.feature_dict)
            asp_generator.user_power_mode = False
            asp_indices = asp_generator.build_network(net0_dict)
            gen_indices = tf.concat([gen_indices, asp_indices], axis=0)
            gen_indices = tf.cast(gen_indices, dtype=tf.int32)
            # self.generator_slate_score = tf.reduce_prod(cr_generator.generator_score, axis=1)
            self.generator_score = cr_generator.generator_score
        elif self.rerank_type == "cr_diversity":
            # cr: 0, diversity: 1
            action = tf.where(tf.equal(search_page, 0), 0, 1)
            # cr generator
            self.generator = cr_generator = generator_lzd.Model(self.flag, input_item_num=self.item_num,
                                                                candidate_size=self.g_candidate_size, batch_size=1,
                                                                variable_scope=self.cr_generator_scope)
            net0_dict = cr_generator.get_net0_dict(self.feature_dict)
            cr_generator.pbc_diversity = False
            gen_indices = cr_generator.build_network(net0_dict)
            # diversity generator
            self.diversity_generator = diversity_generator = generator_lzd.Model(
                self.flag, input_item_num=self.item_num, candidate_size=self.g_candidate_size, batch_size=1,
                variable_scope=self.diversity_generator_scope)
            net0_dict = diversity_generator.get_net0_dict(self.feature_dict)
            diversity_generator.pbc_diversity = True
            diver_indices = diversity_generator.build_network(net0_dict)
            # choose cr or diversity model
            gen_indices = tf.where(tf.equal(action, 1), diver_indices, gen_indices)
            gen_indices = tf.cast(gen_indices, dtype=tf.int32)
            finalScoreState = tf.where(tf.equal(action, 1), diversity_generator.finalScoreState,
                                       self.generator.finalScoreState)
            # self.generator_slate_score = tf.reduce_prod(finalScoreState, axis=1)
        elif self.rerank_type == "diversity":
            self.generator = generator = generator_lzd.Model(self.flag, input_item_num=self.item_num,
                                                             candidate_size=self.g_candidate_size, batch_size=1,
                                                             variable_scope=self.diversity_generator_scope)
            net0_dict = generator.get_net0_dict(self.feature_dict)
            gen_indices = generator.build_network(net0_dict)
            gen_indices = tf.cast(gen_indices, dtype=tf.int32)
            # self.generator_slate_score = tf.reduce_prod(generator.generator_score, axis=1)
        
        # 5. add original index
        org_indices = tf.reshape(tf.range(0, self.list_size), [1, -1])
        # merge all generator topk list
        self.topk_indices = tf.concat([org_indices, gen_indices], axis=0)
        
        self.topk_feature_dict = self.gen_topk_features(self.feature_dict, self.topk_indices)
        CriticModel = importlib.import_module('models.%s' % self.flag.critic_model['name'])
        self.critic = critic = CriticModel.Model(self.flag, self.list_size)
        net0_dict = critic.get_net0_dict(self.topk_feature_dict)
        critic_score = critic.build_network(net0_dict)
        self.topk_critic_score = tf.reshape(critic_score, [-1, self.list_size])  # [B, N]
        
        # start choose slate
        self.item_pay_gmv = self.feature_dict['item_pay_gmv']
        # cr gap
        self.slate_score = slate_score = self.calc_list_score_by_log(self.topk_critic_score)
        cr_gap = (slate_score[1] - slate_score[0]) / slate_score[0]

        if self.rerank_type in ("cr", "cr_diversity", "page_cr_asp"):
            # cr model
            self.slate_index = tf.where(tf.greater(cr_gap, 0), 1, 0)
        elif self.rerank_type == "asp":
            # asp model
            price_gap = self.get_aov_gap()
            gmv_gap = cr_gap + price_gap
            self.slate_index = tf.where(tf.logical_and(tf.greater(gmv_gap, 0), tf.greater(price_gap, 0)), 1, 0)
        elif self.rerank_type == "asp2":
            # asp model
            price_gap = self.get_aov_gap()
            self.slate_index = tf.where(tf.logical_and(tf.greater(cr_gap, 0), tf.greater(price_gap, 0)), 1, 0)
        elif self.rerank_type == "gmv":
            # asp model
            price_gap = self.get_aov_gap()
            gmv_gap = cr_gap + price_gap
            self.slate_index = tf.where(tf.greater(gmv_gap, 0), 1, 0)
        elif self.rerank_type == "cr_asp":
            cr_i = tf.where(tf.greater(cr_gap, 0), 1, 0)
            cr_gap = (slate_score[2] - slate_score[cr_i]) / slate_score[cr_i]
            self.cr_slate_index = tf.where(tf.greater(cr_gap, 0), 2, cr_i)
            aov = self.get_aov()
            price_gap = (aov[2] - aov[cr_i]) / aov[cr_i]
            
            self.gmv_slate_index = tf.where(tf.logical_and(tf.greater(cr_gap, 0), tf.greater(price_gap, 0)), 2, cr_i)
            self.user_power = tf.reshape(self.feature_dict['user_power'], [-1])[0]
            self.slate_index = tf.where(self.user_power >= 5, self.gmv_slate_index, self.cr_slate_index)
        elif self.rerank_type == "cr_asp2":
            cr_i = tf.where(tf.greater(cr_gap, 0), 1, 0)
            cr_gap = (slate_score[2] - slate_score[cr_i]) / slate_score[cr_i]
            self.cr_slate_index = tf.where(tf.greater(cr_gap, 0), 2, cr_i)
            aov = self.get_aov()
            price_gap = (aov[2] - aov[cr_i]) / aov[cr_i]
            gmv_gap = cr_gap + price_gap
    
            self.gmv_slate_index = tf.where(tf.logical_and(tf.greater(gmv_gap, 0), tf.greater(price_gap, 0)), 2, cr_i)
            self.user_power = tf.reshape(self.feature_dict['user_power'], [-1])[0]
            self.slate_index = tf.where(self.user_power >= 5, self.gmv_slate_index, self.cr_slate_index)
        elif self.rerank_type == "cr_asp3":
            cr_i = tf.where(tf.greater(cr_gap, 0), 1, 0)
            asp_cr_gap = (slate_score[2] - slate_score[cr_i]) / slate_score[cr_i]
            aov = self.get_aov()
            price_gap = (aov[2] - aov[cr_i]) / aov[cr_i]
            
            self.slate_index = tf.where(tf.logical_and(tf.greater(asp_cr_gap, 0), tf.greater(price_gap, 0)), 2, cr_i)
        elif self.rerank_type == "cr_asp_gmv":
            cr_i = tf.where(tf.greater(cr_gap, 0), 1, 0)
            cr_gap = (slate_score[2] - slate_score[cr_i]) / slate_score[cr_i]
            self.cr_slate_index = tf.where(tf.greater(cr_gap, 0), 2, cr_i)
            aov = self.get_aov()
            price_gap = (aov[2] - aov[cr_i]) / aov[cr_i]
            gmv_gap = cr_gap + price_gap
            
            self.slate_index = tf.where(tf.logical_and(tf.greater(gmv_gap, 0), tf.greater(price_gap, 0)), 2, cr_i)
        elif self.rerank_type == "cr_asp_gmv3":
            cr_i = tf.where(tf.greater(cr_gap, 0), 1, 0)
            cr_gap = (slate_score[2] - slate_score[cr_i]) / slate_score[cr_i]
            self.cr_slate_index = tf.where(tf.greater(cr_gap, 0), 2, cr_i)
            aov = self.get_aov()
            price_gap = (aov[2] - aov[cr_i]) / aov[cr_i]
    
            self.slate_index = tf.where(tf.logical_and(tf.greater(cr_gap, -0.01), tf.greater(price_gap, 0)), 2, cr_i)
        elif self.rerank_type == "user_power_asp":
            self.cr_slate_index = tf.where(tf.greater(cr_gap, 0), 1, 0)
            # gmv gap
            price_gap = self.get_aov_gap()
            gmv_gap = cr_gap + price_gap
            self.gmv_slate_index = tf.where(tf.logical_and(tf.greater(gmv_gap, 0), tf.greater(price_gap, 0)), 1, 0)
            self.user_power = tf.reshape(self.feature_dict['user_power'], [-1])[0]
            self.slate_index = tf.where(self.user_power >= 6, self.gmv_slate_index, self.cr_slate_index)
        elif self.rerank_type == "user_power_asp2":
            self.cr_slate_index = tf.where(tf.greater(cr_gap, 0), 1, 0)
            # gmv gap
            price_gap = self.get_aov_gap()
            self.gmv_slate_index = tf.where(tf.logical_and(tf.greater(cr_gap, 0), tf.greater(price_gap, 0)), 1, 0)
            self.user_power = tf.reshape(self.feature_dict['user_power'], [-1])[0]
            self.slate_index = tf.where(self.user_power >= 6, self.gmv_slate_index, self.cr_slate_index)
        else:
            self.slate_index = tf.constant(0)
        
        # is_close_rerank=1 close rerank
        is_close_rerank = tf.reshape(feature_dict['is_close_rerank'], [-1])[0]
        self.slate_index = tf.where(tf.equal(is_close_rerank, 0), self.slate_index, 0)
        model_version = 10.0
        
        # rerank score and maidian
        self.choose_index = tf.reshape(tf.nn.embedding_lookup(self.topk_indices, self.slate_index), [-1, 1])
        self.choose_sritic_score = tf.reshape(tf.nn.embedding_lookup(self.topk_critic_score, self.slate_index), [-1])

        rank_score = tf.reshape(tf.linspace(1.0, 0.1, self.list_size), [-1])
        rank_score = rank_score * 1000000 + 3000000
        self.critic_score = tf.reshape(tf.scatter_nd(self.choose_index, self.choose_sritic_score, [self.item_num]), [-1, 1])
        self.rank_score = tf.reshape(tf.scatter_nd(self.choose_index, rank_score, [self.item_num]), [-1, 1])
        self.generator_score = tf.reshape(tf.scatter_nd(self.choose_index, tf.reshape(self.generator_score, [-1]),
                                                        [self.item_num]), [-1, 1])
        critic_index = tf.range(1, self.list_size + 1, 1)
        self.critic_index = tf.reshape(tf.scatter_nd(self.choose_index, critic_index, [self.item_num]), [-1, 1])
        
        # pad 0 to other item
        self.critic_score = tf.pad(self.critic_score, [[0, input_item_num - self.item_num], [0, 0]])
        self.rank_score = tf.pad(self.rank_score, [[0, input_item_num - self.item_num], [0, 0]])
        self.critic_index = tf.pad(self.critic_index, [[0, input_item_num - self.item_num], [0, 0]])
        self.generator_score = tf.pad(self.generator_score, [[0, input_item_num - self.item_num], [0, 0]])
        
        self.choose_slate_index = self.slate_index + 2
        self.choose_slate_index = tf.where(tf.equal(self.choose_slate_index, 2), 1, self.choose_slate_index)
        self.choose_slate_index = tf.tile(tf.reshape(self.choose_slate_index, [-1, 1]), [input_item_num, 1])
        
        # self.generator_slate_score = tf.tile(tf.reshape(self.generator_slate_score, [-1, 1]), [input_item_num, 1])
        model_version = tf.tile(tf.constant([[model_version]]), [input_item_num, 1])
        
        other_rank_score = tf.reshape(tf.range(2600000., 2600000 - input_item_num, -1.), [-1, 1])
        self.rank_score = tf.where(self.rank_score > 0, self.rank_score, other_rank_score)
        self.rank_score = tf.identity(self.rank_score, "rerank_score")
        # 7. define output.
        with tf.variable_scope(name_or_scope='inference_pivot', reuse=tf.AUTO_REUSE):
            self.slate_score = tf.identity(self.generator_score, name="featureparser_trace_slate_score")
            self.critic_score = tf.identity(self.critic_score, name="featureparser_trace_critic_score")
            self.critic_index = tf.identity(self.critic_index, name="featureparser_trace_critic_index")
            self.critic_index = tf.cast(self.critic_index, dtype=tf.float32)
            self.choose_slate_index = tf.identity(self.choose_slate_index, name="featureparser_trace_choose_list")
            self.choose_slate_index = tf.cast(self.choose_slate_index, dtype=tf.float32)
            self.logits = tf.identity(self.rank_score, name="featureparser_trace_rerank_score")
            model_version = tf.identity(model_version, name="featureparser_trace_rerank_version")
            self.finalscore = tf.identity(
                self.logits + self.critic_score * 0.0 + self.slate_score * 0.0 + model_version * 0.0 +
                self.critic_index * 0.0 + self.choose_slate_index * 0.0, name="rerank_predict")
            self.tagscore = tf.identity(self.finalscore, name="rank_predict")
        # 8. load model
        self.load_variables()
    
    @time_log
    def calc_list_score_by_log(self, score):
        score = tf.clip_by_value(1 - score, 1e-10, 1.)
        score = tf.reduce_sum(tf.log(score), axis=1) * -1.
        return score

    @time_log
    def calc_list_score_by_max(self, score):
        score = tf.reduce_max(score, axis=1)
        return score
    
    @time_log
    def get_aov_gap(self):
        self.itemGmv = tf.nn.embedding_lookup(tf.reshape(self.item_pay_gmv, [-1]), self.topk_indices)
        self.itemGmv = tf.reshape(self.itemGmv, [-1, self.list_size])
        # self.itemGmv = tf.exp(self.itemGmv)
        # self.itemGmv = tf.clip_by_value(self.itemGmv, 0, 600)
        aov = tf.reduce_sum(self.itemGmv, axis=1)
        aov_gap = (aov[1] - aov[0]) / aov[0]
        return aov_gap

    @time_log
    def get_aov(self):
        self.itemGmv = tf.nn.embedding_lookup(tf.reshape(self.item_pay_gmv, [-1]), self.topk_indices)
        self.itemGmv = tf.reshape(self.itemGmv, [-1, self.list_size])
        # self.itemGmv = tf.exp(self.itemGmv)
        # self.itemGmv = tf.clip_by_value(self.itemGmv, 0, 600)
        aov = tf.reduce_sum(self.itemGmv, axis=1)
        return aov
    
    @time_log
    def tile_search_pos(self, feature_dict, input_item_num):
        if "search_pos" in feature_dict:
            feature_dict["search_pos"] = tf.slice(tf.tile(feature_dict["search_pos"], [1, input_item_num]),
                                                  [0, 0], [-1, input_item_num])
        return feature_dict
    
    @time_log
    def get_topn_auction(self, feature_dict, rank_num):
        topn_feature_dict = {}
        for feat_name in feature_dict:
            if feat_name != 'search_pos' and feat_name in self.feature_config and self.feature_config[feat_name]['expression'].startswith("user:"):
                topn_feature_dict[feat_name] = feature_dict[feat_name]
            else:
                topn_feature_dict[feat_name] = tf.slice(feature_dict[feat_name], [0, 0], [-1, rank_num])
        return topn_feature_dict
    
    def gen_topk_features(self, feature_dict, indices):
        topk_feature_dict = {}
        for name, tensor in feature_dict.items():
            # print('name=%s, shape=%s' % (name, tensor.get_shape()))
            if name != 'search_pos' and name in self.feature_config and self.feature_config[name]['expression'].startswith("user:"):
                batch = tf.shape(indices)[0]
                topk_feature_dict[name] = tf.tile(tensor, [batch, 1])
            else:
                values = tf.gather(tensor, indices, axis=1)
                topk_feature_dict[name] = tf.reshape(values, tf.shape(indices))
        return topk_feature_dict
    
    @time_log
    def load_variables(self):
        variables = tf.trainable_variables()
        for v in variables:
            print "variable:", v.name
            if v.name.startswith("generator_network"):
                self.cr_variables.append(v)
            elif v.name.startswith("critic_network"):
                self.critic_variables.append(v)
            elif v.name.startswith("asp_generator_net"):
                self.asp_variables.append(v)
            elif v.name.startswith("diversity_generator_net"):
                self.diversity_variables.append(v)
            elif v.name.startswith("actor_network"):
                self.actor_variables.append(v)
        print("critic network variable number: %d" % len(self.critic_variables))
        print("cr network variable number: %d" % len(self.cr_variables))
        print("asp generator network variable number: %d" % len(self.asp_variables))
        print("diversity generator network variable number: %d" % len(self.diversity_variables))
        print("actor network variable number: %d" % len(self.actor_variables))
        self.critic_saver = tf.train.Saver(self.critic_variables)
        if len(self.cr_variables) > 0:
            self.generator_saver = tf.train.Saver(self.cr_variables)
        if len(self.asp_variables) > 0:
            self.asp_saver = tf.train.Saver(self.asp_variables)
        if len(self.diversity_variables) > 0:
            self.diversity_saver = tf.train.Saver(self.diversity_variables)
        if len(self.actor_variables) > 0:
            self.actor_saver = tf.train.Saver(self.actor_variables)
    
    def format_feature_online(self, features, config, seq_len=10):
        feature_dict = {}
        input_item_num = None
        if not isinstance(config, dict):
            config = get_json_conf_from_file(config)
        feature_conf_dict = get_feature_conf_dict(config)
        for name, tensor in features.items():
            try:
                print("raw:### %s, tensor.dtype = %s, tensor.dense_shape= %s" % (
                    name, tensor.dtype, tensor.get_shape()))
                if isinstance(tensor, tf.SparseTensor):
                    if feature_conf_dict[name]['expression'].startswith("user:") and 'already_split' not in feature_conf_dict[name]:
                        # online use ":"
                        if self.flag.is_online == "false":
                            split_char = feature_conf_dict[name]['delimiter'] if 'delimiter' in feature_conf_dict[name] else "|"
                        else:
                            split_char = ":"
                        s_seq_len = feature_conf_dict[name]['seq_len'] if 'seq_len' in feature_conf_dict[name] else seq_len
                        feature_value = pad_sparse2dense(tensor, default_value=split_char.join(['0'] * s_seq_len),
                                                         pad_len=1)
                        feature_value = tf.reshape(feature_value, [-1])
                        feature_value = tf.string_split(feature_value, split_char, skip_empty=False)
                        tensor = pad_sparse2dense(feature_value, default_value='0', pad_len=s_seq_len)
                    elif 'already_split' in feature_conf_dict[name]:
                        s_seq_len = feature_conf_dict[name]['seq_len'] if 'seq_len' in feature_conf_dict[name] else seq_len
                        tensor = pad_sparse2dense(tensor, default_value='0', pad_len=s_seq_len)
                    else:
                        tensor = tf.sparse_to_dense(tensor.indices, tensor.dense_shape, tensor.values,
                                                    default_value="0")
                
                    if feature_conf_dict[name]['value_type'].lower() == "int":
                        tensor = tf.cast(tf.string_to_number(tensor), tf.int32)
                    elif feature_conf_dict[name]['value_type'].lower() in ["float", "double"]:
                        tensor = tf.string_to_number(tensor)
            
                elif feature_conf_dict[name]['value_type'].lower() == "int":
                    tensor = tf.cast(tensor, tf.int32)
            
                if name == 'search_pos':
                    tensor = tf.reduce_max(tensor, keep_dims=True) / 10
                if name == 'usr_curiosity_vector':
                    tensor = tf.reshape(tensor, [1, 32], name=name + '_format')
                else:
                    tensor = tf.reshape(tensor, [1, -1], name=name + '_format')
                feature_dict[name] = tf.identity(tensor, name='%s_assign' % name)
                if input_item_num is None and not feature_conf_dict[name]['expression'].startswith("user:"):
                    input_item_num = tf.shape(feature_dict[name])[1]
            except:
                traceback.print_exc()
                print("error during format online feature: %s, tensor: %s" % (name, tensor))
        return feature_dict, input_item_num
