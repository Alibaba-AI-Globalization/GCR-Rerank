import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-m', type=str, help='mode, [f2j/j2f]', default=None)
parser.add_argument('-i', type=str, help='input file, [fg.json]', default=None)
parser.add_argument('-o', type=str, help='output file, [fg.json]', default=None)
options = parser.parse_args()


def json2features(in_file, out_file):
    with open(in_file, 'r') as fp:
        fg = json.load(fp)
    
    features = []
    for conf in fg['features']:
        features.append(conf['feature_name'])

    with open(out_file, 'w') as fp:
        for name in features:
            fp.write(name + '\n')


def features2json(in_file, out_file):
    feature_dict = {
        "features": []
    }
    with open(in_file, 'r') as fp:
        for line in fp:
            feature = line.strip()
            c = {
                "feature_type": "id_feature",
                "feature_name": feature,
                "expression": "item:"+feature,
                "need_prefix": False,
                "dtype": "int32",
            }
            feature_dict['features'].append(c)
    
    with open(out_file, 'w') as fp:
        json.dump(feature_dict, fp)


def main():
    assert options.m in ['f2j', 'j2f']
    
    if options.m == 'j2f':
        json2features(options.i, options.o)
    elif options.m == 'f2j':
        features2json(options.i, options.o)
    

if __name__ == '__main__':
    main()
