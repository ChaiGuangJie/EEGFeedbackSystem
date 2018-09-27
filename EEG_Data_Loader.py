import mne
import scipy.io as sio
from mne import Epochs, pick_types, find_events

class EEG_Data_Loader(object):
    def __init__(self):
        self.raw = None
        self.info = None
        self.event_id = dict(left=1, right=2)
        self.tmin, self.tmax = -1, 4.

    def load_train_data(self,filename,format='custom'):
        train_data = []
        if format=='custom':
            train_data = self._load_custom_data(filename)
        elif format=='cnt':
            pass
        elif format=='edf':
            pass

        return train_data

    def load_online_data(self,format='custom'):
        pass

    def _load_custom_data(self,filename):
        matFile = sio.loadmat(filename)
        epoch = matFile['epoch']
        # info = matFile['info']
        self.raw = mne.io.RawArray(epoch, self.info)
        # raw = mne.io.read_raw_cnt(filesToOpen[0],None)
        self.raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')
        events = find_events(self.raw, shortest_event=1, stim_channel='STI 014')
        picks = pick_types(self.info, meg=False, eeg=True, stim=False, eog=False,
                           exclude=['eog', 'stim'])
        epochs = Epochs(self.raw, events, self.event_id, self.tmin, self.tmax, picks=picks, preload=True)
        #todo return train_data,labels  #

    def _load_cnt_data(self):
        pass

    def _load_def_data(self):
        pass