import mne
import scipy.io as sio
from mne import Epochs, pick_types, find_events
import numpy as np


class EEG_Data_Loader():
    def __init__(self):
        self.raw = None
        self.info = None
        self.event_id = dict(left=1, right=2)
        self.tmin, self.tmax = -1, 4.
        self.sfreq = None

    def load_train_data(self, t_winSize, t_stride, filename, format='custom',):
        train_data, labels = [], []
        if format == 'custom':
            train_data, labels = self._load_slide_window_data(
                t_winSize, t_stride, filename)  # self._load_custom_data(filename)
            # train_data,labels = self._load_custom_data(filename)
        elif format == 'cnt':
            pass
        elif format == 'edf':
            pass
        return train_data, labels

    def load_online_data(self, format='custom'):
        pass

    def _load_custom_data(self, filename):
        matFile = sio.loadmat(filename)
        data = matFile['epoch']
        self.info = self._mock_raw_info()
        self.raw = mne.io.RawArray(data, self.info)
        # raw = mne.io.read_raw_cnt(filesToOpen[0],None)
        self.raw.filter(
            7.,
            30.,
            fir_design='firwin',
            skip_by_annotation='edge')
        events = find_events(
            self.raw,
            shortest_event=1,
            stim_channel='STI 014')
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

    def _load_concrete_data(self, filename):
        epochs_train, labels = self._load_custom_data(filename)
        left_data = epochs_train[labels == self.event_id['left']]
        right_data = epochs_train[labels == self.event_id['right']]
        return np.concatenate(
            left_data, axis=1), np.concatenate(
            right_data, axis=1)

    def _load_slide_window_data(self, t_winSize, t_stride, filename):

        left_data, right_data = [],[]
        left_concat_data, right_concat_data = self._load_concrete_data(
            filename)
        print(left_concat_data.shape,right_concat_data.shape)
        winSize = int(t_winSize * self.info['sfreq'])  # 时长*频率
        stride = int(t_stride * self.info['sfreq'])
        maxIndex = min(left_concat_data.shape[1], right_concat_data.shape[1]) - winSize
        index = 0
        if maxIndex < winSize:
            return left_data, right_data
        while index <= maxIndex:
            left_data.append(left_concat_data[:, index:index+winSize])
            right_data.append(right_concat_data[:, index:index+winSize])
            index += stride
        labels = np.array([self.event_id['left']] * \
            len(left_data) + [self.event_id['right']] * len(right_data))
        # np.random.shuffle(labels)
        return np.array(left_data + right_data), labels

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
    dataLoader = EEG_Data_Loader()
    train_data, labels = dataLoader.load_train_data(t_winSize=4,t_stride=4,filename="D:/temp/EEG_DATA/EEGData2/真实左手右脚_30_手动控制.mat")
    print(train_data.shape,labels.shape) #todo 每个trial时长

    # left_data, right_data = dataLoader.load_train_data(t_winSize=4,t_stride=4,
    #     filename="D:/temp/EEG_DATA/EEGData2/真实左手右脚_30_手动控制.mat")
    # print(left_data.shape, right_data.shape)
