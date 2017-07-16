from obspy.signal.trigger import *
from preprocessing import minMaxScale, readOneSac


# classic_sta_lta
def classic_sta_lta_feature(trace, nsta=50, nlta=500):
    samp_rate = trace.stats.sampling_rate
    cft = classic_sta_lta(trace.data, int(nsta * samp_rate), int(nlta * samp_rate))
    return cft


# recursive_sta_lta
def recursive_sta_lta_feature(trace, nsta=50, nlta=500):
    samp_rate = trace.stats.sampling_rate
    cft = recursive_sta_lta(trace.data, int(nsta * samp_rate), int(nlta * samp_rate))
    return cft

# delayed_sta_lta
def delayed_sta_lta_feature(trace, nsta=50, nlta=500):
    samp_rate = trace.stats.sampling_rate
    cft = delayed_sta_lta(trace.data, int(nsta * samp_rate), int(nlta * samp_rate))
    return cft

# z-detect
def z_detect_feature(trace, nsta=10):
    samp_rate = trace.stats.sampling_rate
    cft = z_detect(trace.data, int(nsta * samp_rate))
    return cft


# carl_sta_trig
# Do not scale the original data!
def carl_sta_trig_feature(trace, nsta=10, nlta=50, ratio=0.8, quiet=0.8):
    samp_rate = trace.stats.sampling_rate
    cft = carl_sta_trig(trace.data, int(nsta * samp_rate), int(nlta * samp_rate), ratio, quiet)
    return cft


if __name__ == '__main__':
    trace = readOneSac('../sample/after/SC.XJI.2008133160000.D.00.BHZ.sac')
    trace.data = trace.data[:300000]
    # trace.data = minMaxScale(trace.data)

    cft = delayed_sta_lta_feature(trace)
    print trace.data, cft
    plot_trigger(trace, cft, 3, 2)
