from myClient import ScanClient
import mne
from mne.realtime import RtEpochs
from mne.decoding import FilterEstimator
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from myStims import *
import random
import numpy as np


from psychopy import visual,core,event

class FeedbackPipline():
    def __init__(self):
        tmin, tmax = -0.2, 4.
        self.event_id = dict(left=2, right=3)

        self.scanClientt = ScanClient('10.0.180.151',4000)
        self.info = self.scanClient.get_measurement_info()

        picks = mne.pick_types(self.info, meg=False, eeg=True, eog=False,
                               stim=True, exclude=self.info['bads'])

        self.rt_epochs = RtEpochs(self.scanClient, self.event_id, tmin, tmax,
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
        features = self._createFeatures()
        fs = featureStim(self.win, features=features, dotRaduis=10)  # features=features
        fixation = Fixation(self.win, 10)
        x = Xaxis(self.win, radius=self.win.size[0] / 2.0)
        # y = Yaxis(self.win, radius=self.win.size[1] / 2.0)
        arrow_dict = { 1: RightArrow(self.win, 20), -1: LeftArrow(self.win, 20)}
        countDown = CountDown(self.win)

        #todo 如何更新？(每十次以后用新样本更新csp lda?)
        for i in range(self.nOnlineTrial):
            self.scanClientt.start_sending_data() #开始发送数据
            #self.begin_record()

            fixation.draw(2)

            label = self._createRandomLabel()

            arrow = arrow_dict[label]
            arrow.draw(2)

            self.scanClientt.set_event_trigger(label) #手动打标签
            countDown.draw(slightDraw=False)

            self.scanClientt.stop_sending_data() #暂停发送数据

            x.startDraw()

            new_feature = self.get_new_feature(label)

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
        pass

    def _createRandomLabel(self):
        return  random.choice([-1, 1])

    #todo 注册到myClient作为回调
    def _record_raw_buffer(self,raw_buffer):
        if self.record_array is None:
            self.record_array = raw_buffer
        else:
            self.record_array = np.concatenate((self.record_array, raw_buffer.T), axis=1)

    # def _set_record_flag(self,flag):
    #     self.record_flag = flag
    #
    # def begin_record(self):
    #     self._set_record_flag(True)
    #
    # def end_record(self):
    #     self._set_record_flag(False)

    def get_new_feature(self,label):#todo 新特征如果太大 应缩小到适应屏幕边框
        (epoch, _label) = self.rt_epochs.next(return_event_id=True) #todo 怎么结束
        if label != _label:
            raise RuntimeError('数据读取没有同步!')
        epoch = epoch[np.newaxis, :]
        _start = round(epoch.shape[-1] * 0.7 / 4)
        _stop = round(epoch.shape[-1] * 1.8 / 4)
        epoch_data = self.filt.transform(epoch)[:, :, _start:_stop]
        X_train = self.csp.transform(epoch_data)
        feature_x = self.lda.transform(X_train)
        feature_y = 0
        return (feature_x/self.overall_scale,feature_y,_label)




