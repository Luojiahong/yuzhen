# coding=utf-8

from obspy.signal.trigger import *
from preprocessing import minMaxScale, readOneSac
from sklearn.cross_validation import train_test_split
import pandas as pd
import xgboost as xgb
import os


# 传统sta_lta特征函数cft
def classic_sta_lta_feature(trace, nsta=2, nlta=10):
    samp_rate = trace.stats.sampling_rate
    cft = classic_sta_lta(trace.data, int(nsta * samp_rate), int(nlta * samp_rate))
    return cft


# 递归sta_lta特征函数cft，速度快。
def recursive_sta_lta_feature(trace, nsta=2, nlta=10):
    samp_rate = trace.stats.sampling_rate
    cft = recursive_sta_lta(trace.data, int(nsta * samp_rate), int(nlta * samp_rate))
    return cft


# 延迟sta_lta特征函数cft，效果一般。
def delayed_sta_lta_feature(trace, nsta=2, nlta=10):
    samp_rate = trace.stats.sampling_rate
    cft = delayed_sta_lta(trace.data, int(nsta * samp_rate), int(nlta * samp_rate))
    return cft


# z-detect特征函数cft。
def z_detect_feature(trace, nsta=10.0):
    samp_rate = trace.stats.sampling_rate
    cft = z_detect(trace.data, int(nsta * samp_rate))
    return cft


# carl_sta_trig特征函数cft。
def carl_sta_trig_feature(trace, nsta=2, nlta=10, ratio=0.8, quiet=0.8):
    samp_rate = trace.stats.sampling_rate
    cft = carl_sta_trig(trace.data, int(nsta * samp_rate), int(nlta * samp_rate), ratio, quiet)
    return cft


# !!! 得到样本的features
def get_all_positive_features(dir_add):
    # 正样本features
    p_features = pd.DataFrame()
    s_features = pd.DataFrame()
    # 负样本features
    neg_features = pd.DataFrame()

    # 循环读取dir下的每个trace
    files = os.listdir(dir_add)
    for trace_file in files:
        eachTrainFile = os.path.join('%s/%s' % (dir_add, trace_file))
        trace = readOneSac(eachTrainFile)
        trace_len = len(trace.data)
        trace.data = minMaxScale(trace.data, range=(-100, 100))

        # 传统cft变换
        cft_classic_sta_lta_feature = classic_sta_lta_feature(trace, nsta=0.1, nlta=1)
        cft_recursive_sta_lta_feature = recursive_sta_lta_feature(trace, nsta=0.1, nlta=1)
        cft_delayed_sta_lta_feature = delayed_sta_lta_feature(trace, nsta=0.1, nlta=1)
        cft_z_detect_feature = z_detect_feature(trace, nsta=0.1)
        cft_carl_sta_trig_feature = carl_sta_trig_feature(trace, nsta=0.1, nlta=1)

        # 提取正训练样本点
        relativeStartTime = trace.stats.sac.b
        sampling_rate = trace.stats.sampling_rate
        p_point = int((trace.stats.sac.a - relativeStartTime) * sampling_rate)
        s_point = int((trace.stats.sac.t0 - relativeStartTime) * sampling_rate)
        # 提取正训练样本点（p、s）的各维特征
        if 0 < p_point < trace_len:
            this_p_features = pd.DataFrame({'classic': cft_classic_sta_lta_feature[p_point],
                                            'recursive': cft_recursive_sta_lta_feature[p_point],
                                            'delayed': cft_delayed_sta_lta_feature[p_point],
                                            'z_detect': cft_z_detect_feature[p_point],
                                            'carl': cft_carl_sta_trig_feature[p_point]}, index=[trace_file])
        else:
            this_p_features = pd.DataFrame()
        if 0 < s_point < trace_len:
            this_s_features = pd.DataFrame({'classic': cft_classic_sta_lta_feature[s_point],
                                            'recursive': cft_recursive_sta_lta_feature[s_point],
                                            'delayed': cft_delayed_sta_lta_feature[s_point],
                                            'z_detect': cft_z_detect_feature[s_point],
                                            'carl': cft_carl_sta_trig_feature[s_point]}, index=[trace_file])
        else:
            this_s_features = pd.DataFrame()

        p_features = p_features.append(this_p_features)
        s_features = s_features.append(this_s_features)

        # 提取负训练样本点（p前1秒，s后1秒）
        if 0 < p_point < trace_len and 0 < s_point < trace_len:
            neg_points = [p_point - 1 * sampling_rate, s_point + 1 * sampling_rate]
        else:
            neg_points = []
        # 提取负训练样本点的各维特征
        for neg_point in neg_points:
            if neg_point:
                this_neg_features = pd.DataFrame({'classic': cft_classic_sta_lta_feature[neg_point],
                                                  'recursive': cft_recursive_sta_lta_feature[neg_point],
                                                  'delayed': cft_delayed_sta_lta_feature[neg_point],
                                                  'z_detect': cft_z_detect_feature[neg_point],
                                                  'carl': cft_carl_sta_trig_feature[neg_point]}, index=[trace_file])
            else:
                this_neg_features = pd.DataFrame()
            neg_features = neg_features.append(this_neg_features)

    return p_features, s_features, neg_features

