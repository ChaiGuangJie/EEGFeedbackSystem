import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler

from mne import Epochs, pick_types, find_events
from mne.decoding import CSP
import mne



def load_epochs_data(raw, event_id, tmin, tmax, exclude=('eog','stim')):
    raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')
    events = find_events(raw, shortest_event=0, stim_channel='STI 014')
    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude=exclude)
    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=None, preload=True)
    epochs_train = epochs.copy().crop(tmin=0.8, tmax=1.6)#todo
    labels = epochs.events[:, -1] #- 1
    epochs_data_train = epochs_train.get_data()

    return epochs_data_train,labels

def csp_eeg_classify(epochs_data_train,labels):
    scores_list = []
    for c in range(3,10):
        cv = ShuffleSplit(20, test_size=0.2, random_state=32)
        scaler = MinMaxScaler(feature_range=(-0.5, 0.5))#StandardScaler() #MinMaxScaler()#
        lda = LinearDiscriminantAnalysis()
        csp = CSP(n_components=c, reg=None, log=True, norm_trace=False)
        clf = Pipeline([('CSP', csp),('SCALER',scaler), ('LDA', lda)])
        scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1)
        class_balance = np.mean(labels == labels[0])
        class_balance = max(class_balance, 1. - class_balance)
        scores_list.append(round(float(np.mean(scores)),2))

    return scores_list,class_balance #"Classification accuracy: %f / Chance level: %f" % (np.mean(scores),class_balance)

def standard_load(subject,runType=1):
    '''
        ----------
        subject : int
            The subject to use. Can be in the range of 1-109 (inclusive).
        runType : int | list of int
            The runs to use. The runs correspond to:

            =========  ===================================
            run        task
            =========  ===================================
            1          Baseline, eyes open
            2          Baseline, eyes closed
            3, 7, 11   Motor execution: left vs right hand   #runType = 1
            4, 8, 12   Motor imagery: left vs right hand     #runType = 2
            5, 9, 13   Motor execution: hands vs feet        #runType = 3
            6, 10, 14  Motor imagery: hands vs feet          #runType = 4
            =========  ===================================
    '''
    from mne.io import concatenate_raws, read_raw_edf
    from mne.datasets import eegbci
    tmin, tmax = -1., 4.
    event_id = dict(hands=2, feet=3)
    runs = {
        1:[3, 7, 11],
        2:[4, 8, 12],
        3:[5, 9, 13],
        4:[6, 10, 14]
    }

    raw_fnames = eegbci.load_data(subject, runs[runType],path="D:/Project/python/EEG/mne/mne_data")
    raw_files = [read_raw_edf(f, preload=True, stim_channel='auto') for f in
                 raw_fnames]
    raw = concatenate_raws(raw_files)

    # strip channel names of "." characters
    raw.rename_channels(lambda x: x.strip('.'))

    return raw,event_id,tmin,tmax


def load_my_data():
    import scipy.io as sio
    ch_names = ['FP1', 'FPZ', 'FP2','AF3', 'AF4','F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
                'FT7','FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6','FT8','T7',
                'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'HEO',
                'TP7','CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
                'TP8','M2',
                'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
                'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
                'CB1','O1', 'OZ', 'O2', 'CB2','VEO',
                'EMG1', 'EMG2','STI 014']
    ch_types = ['eeg'] * 67
    ch_types.append('stim')
    # ch_types[32] = 'eog'
    # ch_types[64] = 'eog'
    ch_types[42] = 'eog'
    ch_types[65] = 'eog'
    ch_types[66] = 'eog'

    nRate = 1000
    info = mne.create_info(ch_names, nRate, ch_types)
    # info['bads'].append('eog')


    matFile = sio.loadmat("D:/temp/EEG_DATA\EEGData/四秒内任意次真实左肘下右肘上_20trial_自动控制_样本均匀.mat")
    epoch = matFile['epoch']
    # info = matFile['info']
    # epoch = np.concatenate((epoch, epoch[-1:, :]), axis=0)
    raw = mne.io.RawArray(epoch, info)
    event_id = dict(left=1, right=2)
    tmin, tmax = -1., 4.
    # raw.resample(400)
    return raw,event_id,tmin,tmax

if __name__=='__main__':

    # raw,event_id,tmin,tmax = standard_load(subject=4,runType=1)
    raw, event_id, tmin, tmax = load_my_data()
    epochs_data_train, labels = load_epochs_data(raw,event_id,tmin,tmax)

    r = csp_eeg_classify(epochs_data_train,labels)

    print(r)