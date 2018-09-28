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
import random,time
import numpy as np
import scipy.io as sio
from psychopy.gui import fileSaveDlg,fileOpenDlg
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

from myEpochs import _process_raw_buffer
from psychopy import visual,core,event
from sklearn.pipeline import Pipeline

class FeedbackPipline():
    def __init__(self,host = '10.0.181.150',port = 4000,size = (800,600)):
        self.tmin, self.tmax = -1, 4.
        self.event_id = dict(left=1, right=2)

        self.scanClient = ScanClient(host,port)

        self.info = self.scanClient.get_measurement_info()

        self.scanClient.register_receive_callback(self._record_raw_buffer)

        picks = mne.pick_types(self.info, meg=False, eeg=True, eog=False,
                               stim=True, exclude=['eog','stim'])

        self.rt_epochs = RtEpochs(self.scanClient, event_id = list(self.event_id.values()), tmin=self.tmin, tmax = self.tmax,
                             picks=picks, stim_channel='STI 014', isi_max=60 * 60.)
        # self.rt_epochs._process_raw_buffer = _process_raw_buffer #todo 是否可行？
        # self.data_win_size = 801
        self.filter = FilterEstimator(self.info, 7, 30, fir_design='firwin')
        self.csp = CSP(n_components=5, reg=None, log=True, norm_trace=False)
        self.lda = LinearDiscriminantAnalysis()
        self.tsne = TSNE(n_components=2, random_state=0)
        self.scaler_befor_lda = MinMaxScaler(feature_range=(-1, 1))
        self.scaler_after_lda = MinMaxScaler(feature_range=(-1, 1)) #是否需要？

        self.clf =None # Pipeline([('CSP', self.csp),('SCALER_BEFOR',self.scaler_befor_lda),('LDA', self.lda)])
        # self.features = None #保存所有特征点

        self.win = visual.Window(size)
        event.globalKeys.add(key='escape', func=self.quit, name='quit')
        event.globalKeys.add(key='s',modifiers=['ctrl'],func=self.save_raw_file,name='name')

        self.nOfflineTrial = 30
        self.nOnlineTrial = 20

        self.record_array = None
        # self.record_flag = False

        self.overall_scale = 1

        self.saveFileName = None

        self.OfflineScores = 0
        self.OfflineClassBalance = 0

        self.y_offset = 0.35

        # self.record_array_index = -1

    def save_raw_file(self,saveAtMoment=False):

        self.scanClient.stop_sending_data()
        default_name = time.strftime("eeg_epoch_%Y_%m_%d_%Hh%Mm%Ss.mat", time.localtime())
        fullPath = fileSaveDlg(initFilePath="D:\\temp\\EEG_DATA",initFileName=default_name,
                    prompt='保存 EEG Epoch 数据',allowed="Matlab file (*.mat)")
        self.saveFileName = fullPath
        if saveAtMoment:
            self._save_raw_file(fullPath)
        # self.scanClient.start_sending_data()
    def _save_raw_file(self,fullFileName):
        if fullFileName is not None: #todo 应该要等到全部缓存数据都保存进self.record_array再写入文件
            sio.savemat(fullFileName,{'epoch':self.record_array,'info':self.scanClient.basicInfo})
            print(fullFileName,'已保存')

    def run_offline(self):
        fixation = Fixation(self.win, 60)
        arrow_dict = {'right': RightArrow(self.win, 60), 'left': LeftArrow(self.win, 60)}
        countDown = CountDown(self.win, duration=4)
        self.scanClient.start_receive_thread(self.info['nchan'])
        # todo 如何更新？(每十次以后用新样本更新csp lda?)
        DrawTextStim(self.win, "请选择文件保存位置")
        self.save_raw_file()
        WaitOneKeyPress(self.win,'space','按空格键开始')
        for i in range(self.nOfflineTrial):

            self.scanClient.start_sending_data()  # 开始发送数据
            # self.begin_record()
            fixation.draw(2)

            orien, label = self._createRandomLabel()

            arrow = arrow_dict[orien]
            # self.scanClient.set_event_trigger(label)
            arrow.draw(2)

            self.scanClient.set_event_trigger(label)  # 手动打标签

            countDown.draw(slightDraw=True)

            self.scanClient.stop_sending_data()  # 暂停发送数据

            WaitOneKeyPress(self.win,'right','按 → 键继续')
            # arrow.endDraw()
        DrawTextStim(self.win, "实验结束，正在保存文件")
        if self.saveFileName is not None:
            self._save_raw_file(self.saveFileName)
        else:
            self.save_raw_file()

        #self.quit() #
        return
        #先离线采集几组数据并保存

    def run_online(self):
        features = self._create_features_from_offline_data()#[(0.5,0.5,1)]#
        if len(features)==0:return
        self.rt_epochs.start()
        fs = BulletFeaturesStim(self.win,features=features,dotRaduis=20,dt=0.001) # 通过dt控制小球发射速度
        fixation = Fixation(self.win, 30)
        # x = Xaxis(self.win, radius=self.win.size[0] / 2.0,y=0.3*self.win.size[1])
        # y = Yaxis(self.win, radius=self.win.size[1] / 2.0)
        arrow_dict = {'right': RightArrow(self.win, 60), 'left': LeftArrow(self.win, 60)}
        # countDown = CountDown(self.win,duration=4)
        # targetWin,(centerX,centerY) = TargetWindow(self.win)
        # bullet = Bullet(self.win,dotRaduis=30)
        #todo 如何更新？(每十次以后用新样本更新csp lda?)
        DrawTextStim(self.win,"请选择文件保存位置")
        self.save_raw_file()
        WaitOneKeyPress(self.win,'space','按空格键开始')
        nTrial = self.nOnlineTrial
        while nTrial>0:
            self.scanClient.start_sending_data() #开始发送数据
            #self.begin_record()
            fixation.draw(2)
            orien,label = self._createRandomLabel()
            arrow = arrow_dict[orien]
            # self.scanClient.set_event_trigger(label)
            arrow.draw(2)
            self.scanClient.set_event_trigger(label) #手动打标签
###########################################################################
            # countDown.draw(slightDraw=True)#替换成小球连续发射
            # new_feature = self.get_new_feature(label)
            # x.startDraw()
            # fixation.startDraw()
            # self.bulletFeedback(fs,1,label)
            fs.start_fire_bullet(self.getData, label, delay=4, interval=0.5, duration=4) #通过参数控制出球数量与时间间隔
            self.scanClient.set_event_trigger(None) #停止打标签
            self.scanClient.stop_sending_data() #暂停发送数据

            # print(new_feature)
            # fs.drawNewFeature(new_feature)
#############################################################################
            fs.startDrawAllFeatures(gradients=True)

            #self.end_record()
            allKeys = event.waitKeys(keyList=['left', 'right'])
            for thisKey in allKeys:
                if thisKey == 'left':
                    fs.removeLastFeature()
                elif thisKey == 'right':
                    nTrial = nTrial-1
                    break

            fs.endDrawAllFeatures()
            # arrow.endDraw()
            # x.endDraw()
            # y.endDraw()
            # fixation.endDraw()

        DrawTextStim(self.win, "实验结束，正在保存文件")
        if self.saveFileName is not None:
            self._save_raw_file(self.saveFileName)
        else:
            self.save_raw_file()

        #self.quit()
        return

    def run(self):

        self.saveFileName = None
        stimText = "1. 离线采集数据 \n\n2. 在线测试 \n\nEsc 键退出"
        DrawTextStim(self.win, stimText)
        allKeys = event.waitKeys(keyList=['1', '2'])
        for thisKey in allKeys:
            if thisKey == '1':
                self.run_offline()
            elif thisKey == '2':
                self.run_online()

        self.quit()

    def _createFeatures(self):
        #读取离线数据并生成features
        features = []
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
        epochs_train = epochs.copy().crop(tmin=1.5, tmax=2.5) #todo 应该作为全局变量？
        epochs_data_train = epochs_train.get_data()
        X_train = self.csp.fit_transform(epochs_data_train, np.array(label_list)) #todo 离线和在线时的epochs_data_train shape需要一致
        self.lda.fit(X_train, np.array(label_list))
        x = self.lda.transform(X_train)
        self.overall_scale = x.max(axis=0) if x.max(axis=0) > abs(x.min(axis=0)) else abs(x.min(axis=0))
        self.overall_scale = self.overall_scale[0]
        for _x, label in zip(x[:, 0], label_list):
            features.append((_x / self.overall_scale, 0, label))
        return features


    def _create_features_from_offline_data(self):
        # todo 提取特征并展示分类效果 over time
        from sklearn.model_selection import ShuffleSplit, cross_val_score

        import matplotlib.pyplot as plt
        features = []
        DrawTextStim(self.win, "请选择一个离线数据训练分类器")
        filesToOpen = fileOpenDlg(tryFilePath="D:\\temp\\EEG_DATA",prompt='打开 EEG Epoch 数据',allowed="Matlab file (*.mat) ;; CNT file (*.cnt)")
        if filesToOpen is not None:
            DrawTextStim(self.win, "正在训练分类器")
            matFile = sio.loadmat(filesToOpen[0])
            epoch = matFile['epoch']
            # info = matFile['info']
            # _epoch = self.scaler.fit_transform(epoch[:-1,:])
            # epoch = np.concatenate((_epoch,epoch[-1:,:]),axis=0)
            raw = mne.io.RawArray(epoch,self.info)
            # raw = mne.io.read_raw_cnt(filesToOpen[0],None)
            raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')
            events = find_events(raw, shortest_event=1, stim_channel='STI 014')
            picks = pick_types(self.info, meg=False, eeg=True, stim=False, eog=False,
                               exclude=['eog', 'stim'])
            epochs = Epochs(raw, events, self.event_id, self.tmin, self.tmax, picks=picks, preload=True)
            epochs_train = epochs.copy().crop(tmin=0.5, tmax=3.5)#todo 应该去掉？
            # scores = []
            # epochs_data = epochs.get_data()
            epochs_data_train = epochs_train.get_data()
            labels = epochs.events[:, -1]
            cv = ShuffleSplit(10, test_size=0.2, random_state=32)
            # cv_split = cv.split(epochs_data_train)
            scores_list = []
            for c in range(3,10):
                self.csp = CSP(n_components=c, reg=None, log=True, norm_trace=False)
                self.clf = Pipeline([('CSP', self.csp),('LDA', self.lda)]) #,('SCALER_BEFOR',self.scaler_befor_lda)('SCALER_AFTER',self.scaler_after_lda)
                scores = cross_val_score(self.clf, epochs_data_train, labels, cv=cv, n_jobs=1)
                scores_list.append((c,round(float(np.mean(scores)),3)))
                # Printing the results
            class_balance = np.mean(labels == labels[0])
            class_balance = max(class_balance, 1. - class_balance)
            scores_list.sort(key=lambda x: x[1],reverse=True)
            print(scores_list,class_balance)
            # print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
            #                                                           class_balance))
            # self.OfflineScores = np.mean(scores)
            # self.OfflineClassBalance = class_balance
            stimText = str(scores_list) + str(class_balance)
            # stimText = "Classification accuracy: {0}\nChance level: {1}.\n\n空格键继续".format(np.mean(scores),
            #                                                                                      class_balance)
            WaitOneKeyPress(self.win,'space',stimText)
            # DrawTextStim(self.win, stimText)
            # allKeys = event.waitKeys(keyList=['q', 'space'])
            # for thisKey in allKeys:
            #     if thisKey == 'q':
            #         return features
            #     elif thisKey == 'space':
            #         break
            #####################################################
            # self.csp.fit(epochs_data_train,np.array(labels))
            # X_train = self.csp.transform(epochs_data_train)
            # # self.tsne.fit(X_train, np.array(labels))
            # # x = self.tsne.fit_transform(X_train)
            # self.scaler_befor_lda.fit(X_train)
            # X_train = self.scaler_befor_lda.transform(X_train)
            #
            # self.lda.fit(X_train, np.array(labels))
            #
            # self.scaler_after_lda.fit(X_train)
            # X_train = self.scaler_after_lda.transform(X_train)
            #
            # x = self.lda.transform(X_train)
            ############################################################
            # print(scores_list[0][0])
            self.csp = CSP(n_components=scores_list[0][0], reg=None, log=True, norm_trace=False)
            self.clf = Pipeline([('CSP', self.csp),('LDA', self.lda)])#('SCALER_BEFOR',self.scaler_befor_lda)
            clf_result = self.clf.fit_transform(epochs_data_train,labels)
            #todo clf_result的均值是0？
            print(clf_result)
            # print('#######################')
            clf_result_minmax = self.scaler_after_lda.fit_transform(clf_result)
            print(clf_result_minmax)
            for _x, label in zip(clf_result_minmax, labels):#todo lda二分类只能分为两类
                features.append((_x[0] , 0.7, label)) #两类lda只能映射到一维


            ###############################################################################
            # Look at performance over time
            '''
            sfreq = raw.info['sfreq']
            w_length = int(sfreq * 0.5)  # running classifier: window length
            w_step = int(sfreq * 0.1)  # running classifier: window step size
            w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step)

            scores_windows = []

            for train_idx, test_idx in cv_split:
                y_train, y_test = labels[train_idx], labels[test_idx]

                X_train = self.csp.fit_transform(epochs_data_train[train_idx], y_train)
                # X_test = csp.transform(epochs_data_train[test_idx])

                # fit classifier
                self.lda.fit(X_train, y_train)

                # running classifier: test classifier on sliding window
                score_this_window = []
                for n in w_start:
                    X_test = self.csp.transform(epochs_data[test_idx][:, :, n:(n + w_length)])
                    score_this_window.append(self.lda.score(X_test, y_test))
                scores_windows.append(score_this_window)

            # Plot scores over time
            w_times = (w_start + w_length / 2.) / sfreq + epochs.tmin

            plt.figure()
            plt.plot(w_times, np.mean(scores_windows, 0), label='Score')
            plt.axvline(0, linestyle='--', color='k', label='Onset')
            plt.axhline(0.5, linestyle='-', color='k', label='Chance')
            plt.xlabel('time (s)')
            plt.ylabel('classification accuracy')
            plt.title('Classification score over time')
            plt.legend(loc='lower right')
            plt.show()
            '''
            #self.quit()
        return features


    def _createRandomLabel(self):
        key = random.choice(list(self.event_id.keys()))
        return  key,self.event_id[key]

    #册到myClient作为回调
    def _record_raw_buffer(self,raw_buffer):
        if self.record_array is None:
            self.record_array = raw_buffer
        else:
            self.record_array = np.concatenate((self.record_array, raw_buffer), axis=1)

    # def next(self,windowSize):#shape =(68,samples)
    #     self.record_array_index = self.record_array.shape[1]-1-windowSize
    #     if self.record_array_index>=0:
    #         windowData = self.record_array[:,self.record_array_index:-1]
    #         return windowData
    #     else:
    #         return None

    def _current_win_data(self,winSize):
        if self.record_array.shape[1]>winSize:
            return self.record_array[:,winSize:]
        else:
            return None

    def getData(self,winDuration):
        # print('size:',len(window_data))
        winSize = int(self.info['sfreq'] * winDuration)
        window_data = self._current_win_data(winSize)
        # window_data[-1,1:4] = label
        if window_data is not None:
            raw = mne.io.RawArray(window_data, self.info)
            # raw = mne.io.read_raw_cnt(filesToOpen[0],None)
            raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')
            # events = find_events(raw, shortest_event=1, stim_channel='STI 014')
            picks = pick_types(self.info, meg=False, eeg=True, stim=False, eog=False,
                               exclude=['eog', 'stim'])
            # epochs = Epochs(raw, events, self.event_id, self.tmin, self.tmax, picks=picks, preload=True)
            # epochs_data = epochs.copy().get_data()#.crop(tmin=0.5, tmax=3.5)  # todo 应该去掉？
            epochs_data = raw._data[picks]
            # print('..')
            lda_result =self.clf.transform(epochs_data[np.newaxis, :])#
            print(lda_result)
            scaler_result = self.scaler_after_lda.transform(lda_result)
            print(scaler_result)
            return scaler_result[0][0], 0.7
        #todo None的情况 和 0.7 scaler_result.data?
    # def bulletFeedback(self,bullet,intervalTime,currentLabel,duration=4):
    #     # targetWin.startDraw()
    #     clock = core.Clock()
    #     generateTime = clock.getTime()
    #     while clock.getTime() < duration:
    #         if (clock.getTime() - generateTime) > intervalTime:
    #             window_data = self.next(windowSize=self.data_win_size)
    #             generateTime = clock.getTime()
    #         # if window_data is not None:
    #             window_data = np.delete(window_data,[67,66,65,42],axis=0)
    #
    #             # filter_result = self.filt.transform(scaler_result[np.newaxis,:])
    #             csp_result = self.csp.transform(window_data[np.newaxis,:]) #[0]
    #             scaler_result = self.scaler.transform(csp_result)
    #             lda_result = self.lda.transform(scaler_result)
    #             # scaled_data = self.scaler.transform(lda_result) #scaled_data格式？
    #             bullet.add_new_bullet(lda_result[0][0],0,currentLabel)
    #         bullet.update_bullets(0.01) #参数用来控制速度
    #     while True:
    #         bullet.update_bullets(0.01)
    #         # print('update')
    #         if bullet.bulletList[-1]['arrived']:
    #             bullet.bulletList = []
    #             break
    #     print('allArrived')



    def get_new_feature(self,label):#todo 新特征如果太大 应缩小到适应屏幕边框
        (epoch, _label) = self.rt_epochs.next(return_event_id=True) #todo 怎么结束
        if label != _label:
            raise RuntimeError('数据读取没有同步!')
        epoch = epoch[:-1,:]
        epoch = epoch[np.newaxis, :]
        _start = round(epoch.shape[-1] * 0.7 / 4)
        _stop = round(epoch.shape[-1] * 1.8 / 4)
        epoch_data = self.filter.transform(epoch)[:, :, _start:_stop]
        X_train = self.csp.transform(epoch_data) #todo 跟离线维度一致才能transform
        feature_x = self.scaler_befor_lda.transform(X_train)
        feature_x = self.lda.transform(feature_x) #todo归一化，结果放缩到[-1,1]
        feature_y = 0
        return (feature_x,feature_y,_label)

    def quit(self):
        self.scanClient.unregister_receive_callback(self._record_raw_buffer)
        self.rt_epochs.stop(stop_receive_thread=True, stop_measurement=True)
        self.win.close()
        core.quit()


    def AccTest(self):
        pass
    # def _set_record_flag(self,flag):
    #     self.record_flag = flag
    #
    # def begin_record(self):
    #     self._set_record_flag(True)
    #
    # def end_record(self):
    #     self._set_record_flag(False)

if __name__ == '__main__':
    pipline = FeedbackPipline()
    # pipline = FeedbackPipline(host='127.0.0.1', port=5555)
    # pipline.run_online()
    try:
        pipline.run()
        # pipline.run_online()
    except Exception as e:
        print(e)
        pipline.quit()
    # features = pipline._create_features_from_offline_data()

    # fs = featureStim(pipline.win, features=features, dotRaduis=10)
    # TargetWindow(pipline.win).startDraw()
    # fs.startDrawAllFeatures()
    # WaitOneKeyPress(pipline.win, 'space')
    # print(len(features))
    # pipline.quit()
    # pipline = None
    '''
    try:
        # pipline = FeedbackPipline()
        pipline = FeedbackPipline(host='127.0.0.1', port=5555)
        # pipline.run_online()
        pipline._create_features_from_offline_data()
    except RuntimeError as err:
        print(err)
    finally:
        pipline.quit()
    '''
######################################################################################

    '''
    # pipline = FeedbackPipline(host='127.0.0.1',port = 5555)
    pipline.rt_epochs.start()
    # pipline.scanClient.register_receive_callback(pipline._record_raw_buffer)
    # time.sleep(3)
    pipline.scanClient.start_sending_data()
    features = pipline._createFeatures()
    fs = featureStim(pipline.win, features=features, dotRaduis=10)  # features=features
    while True:
        fs.startDrawAllFeatures()
        #allKeys = event.waitKeys(keyList=['left', 'right'])
        # for thisKey in allKeys:
        #     if thisKey == 'left':
        #         fs.removeLastFeature()
        #     elif thisKey == 'right':
        #         break
        allKeys = event.waitKeys(keyList=['left', 'right','up'])
        for thisKey in allKeys:
            if thisKey == 'left':
                fs.removeLastFeature()
            elif thisKey == 'right':
                break
            elif thisKey == 'up':
                print('top')
                pipline.scanClient.set_event_trigger(1)
        fs.endDrawAllFeatures()
        time.sleep(3)
        # self.end_record()
'''






