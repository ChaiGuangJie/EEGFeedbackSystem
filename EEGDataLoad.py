import scipy.io as sio
import numpy as np

class EEGDataLoad():
    def __init__(self):
        pass

    def _load_mat_data(self,filename):
        matFile = sio.loadmat(filename)
        data = matFile['epoch']
        event_channel = data[-1:, :]
        left_index = np.argwhere(event_channel==1)
        right_index = np.argwhere(event_channel==2)
        return None

if __name__=='__main__':
    eegDataLoad = EEGDataLoad()
    data = eegDataLoad._load_mat_data('D:/temp/EEG_DATA/EEGDATA4/mxl左右手想象20次四秒4bd.mat')
    print(data.shape)