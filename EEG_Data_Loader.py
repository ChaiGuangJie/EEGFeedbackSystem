import mne
import scipy.io as sio
from mne import Epochs, pick_types, find_events
import numpy as np
from scipy import signal

class EEG_Data_Loader():
    def __init__(self,t_winSize, t_stride, filename,format='custom'):

        self.t_winSize = t_winSize
        self.t_stride = t_stride
        self.filename = filename
        self.format = format

        self.originRaw = None

        self.raw = None
        self.info = None
        self.event_id = dict(left=1, right=2)
        self.tmin, self.tmax = -1, 4.
        self.sfreq = None

        self.event = None

    def load_train_data(self):
        train_data, labels = [], []
        if self.format == 'custom':
            train_data, labels = self._load_slide_window_data(
                self.t_winSize, self.t_stride, fileter=False)  # self._load_custom_data(filename)
            # train_data,labels = self._load_custom_data(filename)
        elif self.format == 'cnt':
            pass
        elif self.format == 'edf':
            pass
        return train_data, labels

    def load_mock_online_data(self):
        from scipy import  signal
        _mock_online_data, labels = self._load_slide_window_data(
            self.t_winSize, self.t_stride, fileter=False,pick=False)
        # print(self.info['sfreq'])
        mock_online_data = []
        picks = pick_types(
            self.info,
            meg=False,
            eeg=True,
            stim=False,
            eog=False,
            exclude=[
                'eog',
                'stim'])
        _mock_online_data = _mock_online_data[:, picks]
        # raw = mne.io.RawArray(_mock_online_data, self.info)
        # raw.filter(7, 30, fir_design='firwin')
        # t = raw.get_data()[picks]
        lpass = 7
        hpass = 30
        fs = self.info['sfreq']
        filterorder = 3
        filtercutoff = [2 * lpass / fs, 2 * hpass / fs]
        [f_b,f_a] = signal.butter(filterorder,filtercutoff,'bandpass')
        for data in _mock_online_data:
            tempData = np.empty(shape=data.shape)
            for i in range(data.shape[0]):
                tempData[i]= signal.filtfilt(f_b, f_a, data[i])#mne.filter.filter_data(data,self.info['sfreq'],7,30)
            mock_online_data.append(tempData)

        # mock_online_data = mne.filter.filter_data(_mock_online_data,self.info['sfreq'],7,30)
        # return _mock_online_data,labels
        return np.array(mock_online_data),labels
        # return t,labels

    def load_online_data(self):
        pass

    def myfilter(self,data):
        filter_data = []
        lpass = 7
        hpass = 30
        fs = 1000
        filterorder = 3
        filtercutoff = [2 * lpass / fs, 2 * hpass / fs]
        [f_b, f_a] = signal.butter(filterorder, filtercutoff, 'bandpass')
        for dt in data:
            filter_data.append(signal.filtfilt(f_b, f_a, dt))
        return np.array(filter_data)

    def _load_custom_data(self,filter,pick):
        matFile = sio.loadmat(self.filename)
        data = matFile['epoch']
        self.info = self._mock_raw_info()
        ###########################先滤波
        # event_channel = data[-1:,:]
        # # _data = mne.filter.filter_data(data[:-1,:],self.info['sfreq'],7,30)
        # _data = self.myfilter(data[:-1,:])
        # data = np.concatenate((_data,event_channel),axis=0)
        ###########################
        self.raw = mne.io.RawArray(data, self.info)
        # raw = mne.io.read_raw_cnt(filesToOpen[0],None)
        if filter:
            self.raw.filter(
                7.,
                30.,
                fir_design='firwin')
        events = find_events(
            self.raw,
            shortest_event=1,
            stim_channel='STI 014')
        picks = None
        if pick:
            picks = pick_types(
                self.info,
                meg=False,
                eeg=True,
                stim=False,
                eog=False,
                exclude=[
                    'eog',
                    'stim'])
        epochs = Epochs(
            self.raw,
            events,
            self.event_id,
            self.tmin,
            self.tmax,
            picks=picks,
            preload=True)
        labels = epochs.events[:, -1]
        epochs_train = epochs.copy().crop(tmin=0.5, tmax=3.5).get_data()  #
        return epochs_train, labels

    def _load_concrete_data(self,filter,pick):
        epochs_train, labels = self._load_custom_data(filter,pick)
        left_data = epochs_train[labels == self.event_id['left']]
        right_data = epochs_train[labels == self.event_id['right']]
        return np.concatenate(
            left_data, axis=1), np.concatenate(
            right_data, axis=1)

    def load_concrete_data(self,data):
        left_data = data[labels == self.event_id['left']]
        right_data = data[labels == self.event_id['right']]
        return np.concatenate(
            left_data, axis=1), np.concatenate(
            right_data, axis=1)

    def _load_slide_window_data(self, t_winSize, t_stride, fileter=True,pick = True):
        all_data,all_labels  = [],[]
        left_data, right_data = [],[]
        left_concat_data, right_concat_data = self._load_concrete_data(filter=fileter,pick=pick)
        print(left_concat_data.shape,right_concat_data.shape)
        #########################################################
        # left_concat_data = mne.filter.filter_data(left_concat_data,self.info['sfreq'],7,30)
        # right_concat_data = mne.filter.filter_data(right_concat_data,self.info['sfreq'],7,30)
        ########################################################
        winSize = int(t_winSize * self.info['sfreq'])  # 时长*频率
        stride = int(t_stride * self.info['sfreq'])
        maxIndex = min(left_concat_data.shape[1], right_concat_data.shape[1]) - winSize
        index = 0
        if maxIndex < winSize:
            return left_data, right_data
        while index <= maxIndex:
            all_data.append(left_concat_data[:, index:index+winSize])
            all_labels.append(self.event_id['left'])
            # left_data.append(left_concat_data[:, index:index+winSize])
            all_data.append(right_concat_data[:, index:index+winSize])
            all_labels.append(self.event_id['right'])
            # right_data.append(right_concat_data[:, index:index+winSize])
            index += stride
        # labels = np.array([self.event_id['left']] * \
        #     len(left_data) + [self.event_id['right']] * len(right_data))
        # np.random.shuffle(labels)
        # return np.array(left_data + right_data), labels
        return np.array(all_data),np.array(all_labels)

    def _mock_raw_info(self):
        ch_names = ['FP1', 'FPZ', 'FP2',
                    'AF3', 'AF4',
                    'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
                    'FT7',
                    'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6',
                    'FT8',
                    'T7',
                    'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6',
                    'T8',
                    'HEO',
                    'TP7',
                    'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
                    'TP8',
                    'M2',
                    'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
                    'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
                    'CB1',
                    'O1', 'OZ', 'O2',
                    'CB2',
                    'VEO',
                    'EMG1', 'EMG2',
                    'STI 014']
        ch_types = ['eeg'] * 67
        ch_types.append('stim')
        # ch_types[32] = 'eog'
        # ch_types[64] = 'eog'
        ch_types[42] = 'eog'
        ch_types[65] = 'eog'
        ch_types[66] = 'eog'
        nRate = 1000
        return mne.create_info(ch_names, nRate, ch_types)

    def _load_cnt_data(self):
        pass

    def _load_def_data(self):
        pass



if __name__ == "__main__":
    dataLoader = EEG_Data_Loader(t_winSize=3,t_stride=1,filename="D:/temp/EEG_DATA/EEGData2/真实左手右脚_30_手动控制.mat")
    # train_data, labels = dataLoader.load_train_data()
    # print(train_data.shape,labels.shape) #todo 每个trial时长
    mock_online_data,labels = dataLoader.load_mock_online_data()#.load_train_data()#
    print(mock_online_data.shape,labels)
    # left_data, right_data = dataLoader.load_train_data(t_winSize=4,t_stride=4,
    #     filename="D:/temp/EEG_DATA/EEGData2/真实左手右脚_30_手动控制.mat")
    # print(left_data.shape, right_data.shape)
