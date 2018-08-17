from myClient import ScanClient
import mne
from mne.realtime import RtEpochs
from mne.decoding import FilterEstimator
from mne.decoding import CSP

from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne import Epochs, pick_types, find_events

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from myStims import *
import random
import numpy as np


from psychopy import visual,core,event

class FeedbackPipline():
    def __init__(self):
        self.tmin, self.tmax = -0.2, 2.
        self.event_id = dict(left=2, right=3)

        self.scanClient = ScanClient('10.0.180.151',4000)

        self.info = self.scanClient.get_measurement_info()

        picks = mne.pick_types(self.info, meg=False, eeg=True, eog=False,
                               stim=True, exclude=self.info['bads'])

        self.rt_epochs = RtEpochs(self.scanClient, self.event_id, self.tmin, self.tmax,
                             picks=picks, stim_channel='STI 014', isi_max=60 * 60.)

        self.filt = FilterEstimator(self.info, 12, 24, filter_length='auto', fir_design='firwin')
        self.csp = CSP(n_components=5, reg=None, log=True, norm_trace=False)
        self.lda = LinearDiscriminantAnalysis(n_components=1)

        # self.features = None #保存所有特征点

        self.win = visual.Window([1000, 800])
        event.globalKeys.add(key='escape', func=core.quit, name='esc')

        self.nOfflineTrial = 20
        self.nOnlineTrial = 10

        self.record_array = None
        # self.record_flag = False

        self.overall_scale = 1

    def _save_raw_file(self):
        pass #todo 绑定全局键

    def run_offline(self):
        pass #todo 先离线采集几组数据并保存

    def run_online(self):
        self.rt_epochs.start()

        #todo _createFeatures()里面必须要csp.fit lda.fit
        features = self._createFeatures()
        fs = featureStim(self.win, features=features, dotRaduis=10)  # features=features
        fixation = Fixation(self.win, 10)
        x = Xaxis(self.win, radius=self.win.size[0] / 2.0)
        # y = Yaxis(self.win, radius=self.win.size[1] / 2.0)
        arrow_dict = {'right': RightArrow(self.win, 20), 'left': LeftArrow(self.win, 20)}
        countDown = CountDown(self.win)

        #todo 如何更新？(每十次以后用新样本更新csp lda?)
        for i in range(self.nOnlineTrial):
            self.scanClient.start_sending_data() #开始发送数据
            #self.begin_record()

            fixation.draw(2)

            orien,label = self._createRandomLabel()

            arrow = arrow_dict[orien]
            # self.scanClient.set_event_trigger(label)
            arrow.draw(2)

            self.scanClient.set_event_trigger(label) #手动打标签
            countDown.draw(duration=0.12,slightDraw=False)

            new_feature = self.get_new_feature(label)
            self.scanClient.stop_sending_data() #暂停发送数据

            x.startDraw()

            print(new_feature)

            fs.drawNewFeature(new_feature)

            fs.endDrawAllFeatures()

            #self.end_record()
            allKeys = event.waitKeys(keyList=['left', 'right'])
            for thisKey in allKeys:
                if thisKey == 'left':
                    fs.removeLastFeature()
                elif thisKey == 'right':
                    break

            fs.endDrawAllFeatures()
            arrow.endDraw()
            x.endDraw()
            # y.endDraw()
            fixation.endDraw()


    def run(self):

        pass #提示界面 选择离线Or在线数据采集

    def _createFeatures(self):
        #读取离线数据并生成features
        features = []
        # for i in range(20):
        #     # label = random.choice([-1, 1])
        #     label = random.choice(list(self.event_id.values()))
        #     mean = np.mean(list(self.event_id.values()))
        #     x = random.gauss(label - mean, 0.2)
        #     # x = random.uniform(-1, 1)
        #     y = 0  # random.uniform(-1, 1)
        #     features.append((x, y, label))
        subject = 2
        runs = [4, 8, 12]  # Motor imagery: left vs right hand

        raw_fnames = eegbci.load_data(subject, runs, path="D:/Project/python/EEG/mne/mne_data")
        raw_files = [read_raw_edf(f, preload=True, stim_channel='auto') for f in
                     raw_fnames]
        raw = concatenate_raws(raw_files)

        # strip channel names of "." characters
        raw.rename_channels(lambda x: x.strip('.'))

        # Apply band-pass filter
        # raw.filter(12., 24., fir_design='firwin', skip_by_annotation='edge')

        events = find_events(raw, shortest_event=0, stim_channel='STI 014')

        picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                           exclude='bads')

        # Read epochs (train will be done only between 1 and 2s)
        # Testing will be done with a running classifier
        epochs = Epochs(raw, events, self.event_id, self.tmin, self.tmax, proj=True, picks=picks,
                        baseline=None, preload=True)
        label_list = epochs.events[:, -1]
        epochs_train = epochs.copy().crop(tmin=0.8, tmax=1.6)
        epochs_data_train = epochs_train.get_data()
        X_train = self.csp.fit_transform(epochs_data_train, np.array(label_list)) #todo 离线和在线时的epochs_data_train shape需要一致？
        self.lda.fit(X_train, np.array(label_list))
        x = self.lda.transform(X_train)
        self.overall_scale = x.max(axis=0) if x.max(axis=0) > abs(x.min(axis=0)) else abs(x.min(axis=0))
        self.overall_scale = self.overall_scale[0]
        for _x, label in zip(x[:, 0], label_list):
            features.append((_x / self.overall_scale, 0, label))
        return features
        # pass

    def _createRandomLabel(self):
        key = random.choice(list(self.event_id.keys()))
        return  key,self.event_id[key]

    #todo 注册到myClient作为回调
    def _record_raw_buffer(self,raw_buffer):
        if self.record_array is None:
            self.record_array = raw_buffer
        else:
            self.record_array = np.concatenate((self.record_array, raw_buffer.T), axis=1)


    def get_new_feature(self,label):#todo 新特征如果太大 应缩小到适应屏幕边框
        (epoch, _label) = self.rt_epochs.next(return_event_id=True) #todo 怎么结束
        if label != _label:
            raise RuntimeError('数据读取没有同步!')
        epoch = epoch[np.newaxis, :]
        _start = round(epoch.shape[-1] * 0.7 / 4)
        _stop = round(epoch.shape[-1] * 1.8 / 4)
        epoch_data = self.filt.transform(epoch)[:, :, _start:_stop]
        X_train = self.csp.transform(epoch_data) #todo 跟离线维度一致才能transform
        feature_x = self.lda.transform(X_train)
        feature_y = 0
        return (feature_x/self.overall_scale,feature_y,_label)


    # def _set_record_flag(self,flag):
    #     self.record_flag = flag
    #
    # def begin_record(self):
    #     self._set_record_flag(True)
    #
    # def end_record(self):
    #     self._set_record_flag(False)

if __name__ == '__main__':
    try:
        pipline = FeedbackPipline()
        pipline.run_online()
    finally:
        pipline.rt_epochs.stop(stop_receive_thread=True,stop_measurement=True)




