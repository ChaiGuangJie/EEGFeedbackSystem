from psychopy import visual,core,event
import random
from myStims import *
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.realtime import MockRtClient, RtEpochs
from mne.datasets import sample
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne.decoding import Vectorizer, FilterEstimator  # noqa
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP

print(__doc__)

record_flag = False
record_array = None

tmin, tmax = -0.2 , 4.
event_id = dict(left = 2, right = 3)
subject = 2
# runs = [6, 10, 14]  # motor imagery: hands vs feet
runs = [4, 8, 12]  # Motor imagery: left vs right hand

raw_fnames = eegbci.load_data(subject, runs,path="D:/Project/python/EEG/mne/mne_data")
raw_files = [read_raw_edf(f, preload=True, stim_channel='auto') for f in
             raw_fnames]
raw = concatenate_raws(raw_files)

# strip channel names of "." characters
raw.rename_channels(lambda x: x.strip('.'))


tr_percent = 60  # Training percentage
min_trials = 20  # minimum trials after which decoding should start

# select gradiometers
picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False,
                       stim=True, exclude=raw.info['bads'])

# create the mock-client object
rt_client = MockRtClient(raw)

# create the real-time epochs object
rt_epochs = RtEpochs(rt_client, event_id, tmin, tmax, picks=picks, decim=1, baseline=None,
                     isi_max=60*60.)

# start the acquisition
rt_epochs.start()

# send raw buffers
rt_client.send_data(rt_epochs, picks, tmin=0, tmax=369., buffer_size=40)


filt = FilterEstimator(raw.info, 12, 24,  filter_length='auto',fir_design='firwin')
lda = LinearDiscriminantAnalysis(n_components = 1)
csp = CSP(n_components=5, reg=None, log=True, norm_trace=False)
#
win = visual.Window([1000, 800])
event.globalKeys.add(key='escape', func=core.quit, name='esc')

trial = 1
overall_scale = 0

#绑定全局建用于保存采集数据
def save_raw_buffer():
    pass

#注册到myClient作为回调
def record_raw_buffer(raw_buffer):
    global record_flag,record_array
    if record_flag:
        if record_array is None:
            record_array = raw_buffer
        else:
            record_array = np.concatenate((record_array,raw_buffer.T),axis=1)

rt_client.register_receive_callback(record_raw_buffer)

def fix_label(_label):
    return int((_label-2.5)*2)

def createFeatures(rt_epochs,filt,csp,lda,n_features = 40):
    global overall_scale
    features = []
    epoch_list = []
    label_list = []
    for i in range(n_features):
        (epoch, _label) = rt_epochs.next(return_event_id=True)
        epoch = epoch[np.newaxis, :]
        _start = round(epoch.shape[-1] * 0.7 / 4)
        _stop = round(epoch.shape[-1] * 1.8 / 4)
        epoch_data = filt.transform(epoch)[:, :, _start:_stop]
        epoch_list.append(epoch_data)
        label_list.append(fix_label(_label))
    epoch_array = np.concatenate(epoch_list)
    X_train = csp.fit_transform(epoch_array, np.array(label_list))
    lda.fit(X_train,np.array(label_list))
    x = lda.transform(X_train)
    overall_scale = x.max(axis = 0) if x.max(axis = 0) > abs(x.min(axis = 0)) else abs(x.min(axis = 0))

    for _x,label in zip(x[:,0],label_list):
        features.append((_x/overall_scale, 0, label))
    return features

def createRandomArrow(label = None):
    arrowDict = {
        1: RightArrow(win, 20),
        -1: LeftArrow(win, 20)
    }
    if label is not None:
        if label == 1 or label == -1:
            arrow = label
        else:
            raise RuntimeError('label 标签不正确')
            print('label 标签不正确')
    else:
        arrow = random.choice([-1, 1])
    return arrowDict[arrow]

features = createFeatures(rt_epochs,filt,csp,lda)
fs = featureStim(win,features=features,dotRaduis=10) #features=features

fixation = Fixation(win, 10)

while True:
    record_flag = True
    # fixation.startDraw()
    fixation.draw(2)

    (epoch, _label) = rt_epochs.next(return_event_id=True)
    # l = LeftArrow(win,20)
    # l.draw(3)
    # r = RightArrow(win,20)
    # r.draw(5)
    arrow = createRandomArrow(fix_label(_label))
    arrow.draw(2)

    countDown = CountDown(win)
    # transport.run()
    #todo 手动打标签
    countDown.draw(slightDraw=False)
    # transport.pause()
    # todo 是否需要手动取消标签
    record_flag = False
    fixation.startDraw()
    x = Xaxis(win, radius=win.size[0] / 2.0)
    y = Yaxis(win)
    x.startDraw()
    # y.startDraw()
    # y.draw(5)
    # x.endDraw()

    #_label = random.choice([-1, 1])

    epoch = epoch[np.newaxis,:]
    _start = round(epoch.shape[-1] * 0.7 / 4)
    _stop = round(epoch.shape[-1] * 1.8 / 4)
    epoch_data = filt.transform(epoch)[:,:,_start:_stop]
    X_train = csp.transform(epoch_data)
    # X_train = csp.fit_transform(epoch_data,np.array([_label]))
    _x = lda.transform(X_train)
    fs.drawNewFeature((_x[0][0]/overall_scale, 0, fix_label(_label)))  # random.uniform(-1, 1)
    print(_x[0][0]/overall_scale,fix_label(_label)) #todo 样本的分界面是零点？
    fs.startDrawAllFeatures(gradients=True)
    # core.wait(5)
    print('trial ', trial, ' end')
    trial += 1

    event.waitKeys(keyList=['space'])

    fs.endDrawAllFeatures()
    arrow.endDraw()
    x.endDraw()
    # y.endDraw()
    fixation.endDraw()

