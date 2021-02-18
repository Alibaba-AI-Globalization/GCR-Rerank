import sys
import json


def get_feature_mean_std(filename):
    feature_mean_std_dict = {}
    with open(filename, "r") as f:
        for line in f:
            sp = line.split("\t")
            if len(sp) == 7 and sp[1] == 'keyword':
                name = sp[2]
                mean = float(sp[3])
                std = float(sp[4])
                feature_mean_std_dict[name] = (mean, std)
    return feature_mean_std_dict


def set_fg_mean_std(filename):
    with open(filename, 'r') as fp:
        fg_json = json.load(fp)
        for conf in fg_json['preprocess']:
            feature_name = conf['feature_name']
            if 'scale_mean' in conf and feature_name in feature_mean_std_dict:
                conf['scale_mean'] = feature_mean_std_dict[feature_name][0]
            if 'scale_stddev' in conf and feature_name in feature_mean_std_dict:
                conf['scale_stddev'] = feature_mean_std_dict[feature_name][1]
    return fg_json


if __name__ == '__main__':
    # python ./feature_norm.py ${localFeatureStatistic} ${confTemplate} fg_new.json
    feature_statistic = sys.argv[1]
    fg_file = sys.argv[2]
    save_file = sys.argv[3]
    feature_mean_std_dict = get_feature_mean_std(feature_statistic)
    fg_json = set_fg_mean_std(fg_file)
    
    with open(save_file, 'w') as fp:
        json.dump(fg_json, fp)
