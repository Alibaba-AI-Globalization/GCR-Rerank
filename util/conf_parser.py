import os
import json
import sys
from .record import TableRecord


class ConfParser(object):
    def __init__(self, mode, model, conf_file='', experiment='', version='', nation='', data_date='',
                 env='', oss_key='', current=''):
        self.experiment = experiment
        self.version = version
        self.nation = nation
        self.env = env
        self.oss_key = oss_key
        self.mode = mode
        self.current = current
        self.conf_file = conf_file
        self.data_date = data_date
        
        if model == "c":
            self.model = "critic"
        elif model == "g":
            self.model = "generator"
        else:
            self.model = model

        self.replace_dict = {
            '${experiment}': experiment,
            "${model}": self.model,
            '${version}': version
            #'${nation}': nation,
            #'${NATION}': nation.swapcase(),
            #'${space}': 'odps://%s/tables' % env,
            #'${current}': current,
            #"${date}": data_date,
        }

        self.parameters = self.parse()
        self.parameters['mode'] = mode
        self.parameters['model'] = model
        self.used_cfgs = {}

        '''
        # record train and test ODPS-tables
        if mode not in ["export", "local"]:
            table_record = TableRecord(self.parameters['data'][self.model + '_train'],
                                       self.parameters['data'][self.model + '_test'])
            self.table_record = table_record
            table_record.localize()
            
            if self.model == "critic":
                self.parameters['data']['critic_result_table'] = self.parameters['data']['critic_result_table'].replace(
                    '${data}', table_record.sha)
        '''
        pass

    def replace(self, string):
        for old, new in self.replace_dict.items():
            string = string.replace(old, new)
        return string

    def parse(self):
        parameters = {}
        with open(self.conf_file, 'r') as fp:
            _parameters = json.load(fp)
        
        # replace variable configures.
        for block in _parameters.keys():
            if isinstance(_parameters[block], dict):
                block_dict = {}
                for key, value in _parameters[block].items():
                    # note:  `str` or `unicode` in python2
                    block_dict[key] = self.replace(value) if not isinstance(value,
                                                                            (int, float, bool, list, tuple, dict)) else value
                    # block_dict[key] = value
                parameters[block] = block_dict
            else:
                parameters[block] = _parameters[block]
        return parameters

    '''
    def buckets(self):
        bucket_set = set()
        for k, v in self.parameters['path'].items():
            if v.startswith("oss"):
                bucket_set.add(v)
        
        bucket_list = list(bucket_set)
        if len(bucket_list) >= 2:
            ret = ','.join([bucket_list[0] + self.oss_key] + bucket_list[1:])
        elif len(bucket_list) >= 1:
            ret = bucket_list[0] + self.oss_key
        else:
            ret = ''
        return ret
    
    def tables(self):
        model = self.model
        if self.mode in ('train', 'local_train'):
            table_list = list({self.parameters['data'][model + '_train'], self.parameters['data'][model + '_valid']})
        elif self.mode == 'test':
            table_list = list({self.parameters['data'][model + '_test']})
        
        elif self.mode == 'export' or self.mode == 'local':
            table_list = list({self.parameters['data']['critic_test']})
        elif self.mode == 'debug':
            table_list = list({self.parameters['data'][model + '_test']})
        
        ret = ','.join(table_list)
        return ret
    
    def output(self):
        return self.parameters['data'][self.model + '_result_table']
        
    def get_cluster(self, cfg):
        value = self.parameters['distribution'][cfg]
        self.used_cfgs[cfg] = value
        return value
    '''
    
    def to_string(self):
        fmt_str = ""
        removed = ['cluster']
        for name, value in self.parameters.items():
            if name not in removed:
                fmt_str += ('--%s=%s ' % (name, str(value)))
        return fmt_str
