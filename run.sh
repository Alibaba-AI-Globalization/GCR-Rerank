#!/usr/bin/env bash

# train critic
python2.7 run.py --env_str=local_train,critic,conf/param_fs_v1_gmv.json,local,v1
# train generator
python2.7 run.py --env_str=local_train,generator,conf/param_fs_v1_gmv.json,local,v1
# export total rerank model.
python2.7 run.py --env_str=export,rerank,conf/param_fs_v1_gmv.json,local,v1


