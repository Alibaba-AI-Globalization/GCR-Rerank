import json
import sys

# python util/gen_context_features.py conf/fg_v11.json
def get_features(in_file):
    with open(in_file, 'r') as fp:
        fg = json.load(fp)
    
    features = {}
    for conf in fg['features']:
        features[conf['feature_name']] = conf['value_type']
    return features


features = get_features(sys.argv[1])
for name in features:
    vtype = features[name]
    if vtype in ("Int", "String"):
        print("\"" + name + "_same\",")
    else:
        print("\"" + name + "_mean\",")
        print("\"" + name + "_std\",")
        print("\"" + name + "_bias\",")
        print("\"" + name + "_bigger\",")
        print("\"" + name + "_order\",")
