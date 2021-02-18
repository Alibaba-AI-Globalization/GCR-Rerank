#!/bin/sh
ds=$1
nation=$2
echo "set feature normalized as ds="${ds}" nation="${nation}
confTemplate=$3

localFeature2Track='./feature_2_track.dict'
rm -f ${localFeature2Track}
odpscmd -e "use voyager_algo; get resource feature_2_track.dict ${localFeature2Track};"

localFeatureStatistic='./feature_statistic'
rm -f ${localFeatureStatistic}
odpscmd -e "tunnel download voyager_algo.lzd_deep_ltr_feature_static_info_table/ds='${ds}' ${localFeatureStatistic} -fd '\t';"

#deal features for nation
python ./feature_norm.py ${localFeatureStatistic} ${confTemplate} fg_new.json


