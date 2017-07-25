# coding=utf-8

from obspy.signal.trigger import *
from preprocessing import minMaxScale, readOneSac
import pandas as pd
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


def get_all_features(dir):
    p_features = pd.DataFrame()
    s_features = pd.DataFrame()

    files = os.listdir(dir)
    for trace_file in files:
        eachTrainFile = os.path.join('%s/%s' % (dir, trace_file))
        trace = readOneSac(eachTrainFile)
        trace_len = len(trace.data)
        trace.data = minMaxScale(trace.data, range=(-100, 100))

        # 提取正训练样本点
        relativeStartTime = trace.stats.sac.b
        sampling_rate = trace.stats.sampling_rate
        p_point = int((trace.stats.sac.a - relativeStartTime) * sampling_rate)
        s_point = int((trace.stats.sac.t0 - relativeStartTime) * sampling_rate)

        # 提取正训练样本点的各维特征
        cft_classic_sta_lta_feature = classic_sta_lta_feature(trace, nsta=0.1, nlta=1)
        cft_recursive_sta_lta_feature = recursive_sta_lta_feature(trace, nsta=0.1, nlta=1)
        cft_delayed_sta_lta_feature = delayed_sta_lta_feature(trace, nsta=0.1, nlta=1)
        cft_z_detect_feature = z_detect_feature(trace, nsta=0.1)
        cft_carl_sta_trig_feature = carl_sta_trig_feature(trace, nsta=0.1, nlta=1)

        # 做成pandas
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

    return p_features, s_features


if __name__ == '__main__':
    p, s = get_all_features('./example30')
    print p
    print '----'
    print s
