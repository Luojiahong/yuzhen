# coding=utf-8

from obspy.signal.trigger import *
from preprocessing import minMaxScale, readOneSac
import pandas as pd
import xgboost as xgb
import os
# import matplotlib.pyplot as plt


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


# 能量变化率cft。
def energy_change_ratio(trace):
    trace_data = np.concatenate([trace.data, [0]], axis=0)
    extra_trace = np.concatenate([[0], trace.data], axis=0)
    cft = np.array(trace_data) - np.array(extra_trace)
    cft = cft[:-1]
    # for i in range(1, len(trace.data)):
    #     trace.data[i] = trace.data[i] - trace.data[i - 1]
    # cft = trace.data
    return cft


def forward_backward_ratio_point(trace, point, t=1, type='var'):
    sampling_rate = trace.stats.sampling_rate
    length = len(trace.data)
    forward = int(point - t * sampling_rate)
    backward = int(point + t * sampling_rate)
    if type == 'var':
        if forward > 0 and backward < length:
            forward_seq = trace.data[forward: point]
            backward_seq = trace.data[point: backward]
            forward_var = np.var(forward_seq)
            backward_var = np.var(backward_seq)
            if backward_var == 0:
                ratio = 999
            else:
                ratio = forward_var / backward_var
        else:
            ratio = 999
    else:
        if forward > 0 and backward < length:
            forward_seq = trace.data[forward: point]
            backward_seq = trace.data[point: backward]
            forward_mean = np.mean(forward_seq)
            backward_mean = np.mean(backward_seq)
            if backward_mean == 0:
                ratio = 999
            else:
                ratio = forward_mean / backward_mean
        else:
            ratio = 999
    return ratio


def forward_backward_ratio_trace(trace, t=2, type='mean'):
    sampling_rate = trace.stats.sampling_rate
    length = len(trace.data)
    ratio_trace = []
    for point in xrange(0, length):
        forward = int(point - t * sampling_rate)
        backward = int(point + t * sampling_rate)
        if type == 'var':
            if forward > 0 and backward < length:
                forward_seq = trace.data[forward: point]
                backward_seq = trace.data[point: backward]
                forward_var = np.var(forward_seq)
                backward_var = np.var(backward_seq)
                if backward_var == 0:
                    ratio = 999
                else:
                    ratio = forward_var / backward_var
            else:
                ratio = 0
        else:
            if forward > 0 and backward < length:
                forward_seq = trace.data[forward: point]
                backward_seq = trace.data[point: backward]
                forward_mean = np.mean(forward_seq)
                backward_mean = np.mean(backward_seq)
                if backward_mean == 0:
                    ratio = 999
                else:
                    ratio = forward_mean / backward_mean
            else:
                ratio = 0
        ratio_trace.append(ratio)
    return ratio_trace


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
                neg_point = int(neg_point)
                this_neg_features = pd.DataFrame({'classic': cft_classic_sta_lta_feature[neg_point],
                                                  'recursive': cft_recursive_sta_lta_feature[neg_point],
                                                  'delayed': cft_delayed_sta_lta_feature[neg_point],
                                                  'z_detect': cft_z_detect_feature[neg_point],
                                                  'carl': cft_carl_sta_trig_feature[neg_point]}, index=[trace_file])
            else:
                this_neg_features = pd.DataFrame()
            neg_features = neg_features.append(this_neg_features)

    return p_features, s_features, neg_features


# 提取
# 线上online训练集中各点的features
def get_points_features(trace, points):
    trace.data = minMaxScale(trace.data, range=(-100, 100))
    # 传统cft变换
    cft_classic_sta_lta_feature = classic_sta_lta_feature(trace, nsta=0.1, nlta=1)
    cft_recursive_sta_lta_feature = recursive_sta_lta_feature(trace, nsta=0.1, nlta=1)
    cft_delayed_sta_lta_feature = delayed_sta_lta_feature(trace, nsta=0.1, nlta=1)
    cft_z_detect_feature = z_detect_feature(trace, nsta=0.1)
    cft_carl_sta_trig_feature = carl_sta_trig_feature(trace, nsta=0.1, nlta=1)

    points_features = pd.DataFrame()
    for point in points:
        this_features = pd.DataFrame({'classic': cft_classic_sta_lta_feature[point],
                                      'recursive': cft_recursive_sta_lta_feature[point],
                                      'delayed': cft_delayed_sta_lta_feature[point],
                                      'z_detect': cft_z_detect_feature[point],
                                      'carl': cft_carl_sta_trig_feature[point]}, index=[point])
        points_features = points_features.append(this_features)
    return points_features


# if __name__ == '__main__':
#     # 01.JMG.BHE.SAC
#     # 04.JMG.BHE.SAC
#     # 07.QCH.BHE.SAC
#     # 10.PWU.BHZ.SAC
#     # 11.QCH.BHZ.SAC
#     # 17.PWU.BHN.SAC
#     # 22.JMG.BHE.SAC
#     # 26.PWU.BHN.SAC
#     # 28.PWU.BHN.SAC
#     # ./sample/example30/14.PWU.BHN.SAC
#     # ./preliminary/after/GS.WDT.2008212160001.BHN
#     trace = readOneSac('./preliminary/after/XX.JJS.2008240000000.BHN')
#     trace.data = trace.data[1:165000]
#     # trace.plot()
#     # trace.data = energy_change_ratio(trace)
#     # trace.data = minMaxScale(trace.data, range=(-100,100))
#
#     relativeStartTime = trace.stats.sac.b
#     sampling_rate = trace.stats.sampling_rate
#     p_point = int((trace.stats.sac.a - relativeStartTime) * sampling_rate)
#     s_point = int((trace.stats.sac.t0 - relativeStartTime) * sampling_rate)
#     print p_point, s_point
#
#     print 'p', forward_backward_ratio_point(trace, p_point, t=1, type='var')
#     print 's', forward_backward_ratio_point(trace, s_point, t=1, type='var')
#
#     # x, y
#     ratio = forward_backward_ratio_trace(trace, type='var', t=1)
#     x = range(0, len(trace.data))
#
#     # for plot
#     fig1 = plt.subplot(211)
#     plt.plot(x, trace.data)
#     ymin1, ymax1 = fig1.get_ylim()
#     plt.vlines([p_point, s_point], ymin1, ymax1, colors='red')
#
#     fig2 = plt.subplot(212)
#     plt.plot(x, ratio)
#     ymin2, ymax2 = fig2.get_ylim()
#     plt.vlines([p_point, s_point], ymin2, ymax2, colors='red')
#
#     plt.show()
