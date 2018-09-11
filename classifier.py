"""
===========================================================================
Motor imagery decoding from EEG data using the Common Spatial Pattern (CSP)
===========================================================================

Decoding of motor imagery applied to EEG data decomposed using CSP.
Here the classifier is applied to features extracted on CSP filtered signals.

See http://en.wikipedia.org/wiki/Common_spatial_pattern and [1]_. The EEGBCI
dataset is documented in [2]_. The data set is available at PhysioNet [3]_.

References
----------

.. [1] Zoltan J. Koles. The quantitative extraction and topographic mapping
       of the abnormal components in the clinical EEG. Electroencephalography
       and Clinical Neurophysiology, 79(6):440--447, December 1991.
.. [2] Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N.,
       Wolpaw, J.R. (2004) BCI2000: A General-Purpose Brain-Computer Interface
       (BCI) System. IEEE TBME 51(6):1034-1043.
.. [3] Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG,
       Mietus JE, Moody GB, Peng C-K, Stanley HE. (2000) PhysioBank,
       PhysioToolkit, and PhysioNet: Components of a New Research Resource for
       Complex Physiologic Signals. Circulation 101(23):e215-e220.
"""
# Authors: Martin Billinger <martin.billinger@tugraz.at>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

from mne import Epochs, pick_types, find_events
from mne.channels import read_layout
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP
import mne
import scipy.io as sio

print(__doc__)

# #############################################################################
# # Set parameters and read data

# avoid classification of evoked responses by using epochs that start 1s after
# cue onset.
tmin, tmax = -0.2 , 4.
event_id = dict(left = 1, right = 2)
matFile = sio.loadmat("D:/temp/EEG_DATA/EEGData/四秒内任意次真实左肘下右肘上_20trial_自动控制_样本均匀.mat")
event_line = matFile['epoch'][-1,:].reshape(1,-1)
epoch = matFile['epoch'][:-1,:]*0.000001
epoch = np.concatenate((epoch,event_line),axis=0)
info = matFile['info']
nRate = info[0][0][4] #采样率 1000
ch_names = ['FP1', 'FPZ', 'FP2',
                    'AF3', 'AF4',
                    'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
                    'FT7',
                    'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6',
                    'FT8',
                    'T7',
                    'C5', 'C3', 'C1', 'CZ','C2', 'C4', 'C6',
                    'T8',
                    'HEO',
                    'TP7',
                    'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
                    'TP8',
                    'M2',
                    'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
                    'PO7', 'PO5', 'PO3','POZ', 'PO4', 'PO6', 'PO8',
                    'CB1',
                    'O1', 'OZ', 'O2',
                    'CB2',
                    'VEO',
                    'EMG1','EMG2',
                    'STI 014']
ch_types = ['eeg']*67
ch_types.append('stim')
ch_types[32] = 'eog'
ch_types[42] = 'eog'
ch_types[64] = 'eog'
ch_types[65] = 'eog'
ch_types[66] = 'eog'
bads = ['FC3','M2','EMG1','EMG2','VEO','HEO']
raw_info = mne.create_info(ch_names, nRate, ch_types)
raw_info['bads'] = bads
raw = mne.io.RawArray(epoch, raw_info)
# subject = 2
# runs = [6, 10, 14]  # motor imagery: hands vs feet
# runs = [4, 8, 12]  # Motor imagery: left vs right hand


# raw_fnames = eegbci.load_data(subject, runs,path="D:/Project/python/EEG/mne/mne_data")
# raw_files = [read_raw_edf(f, preload=True, stim_channel='auto') for f in
#              raw_fnames]
# raw = concatenate_raws(raw_files)

# strip channel names of "." characters
# raw.rename_channels(lambda x: x.strip('.'))

# Apply band-pass filter
raw.filter(7.,30., fir_design='firwin', skip_by_annotation='edge')

events = find_events(raw, stim_channel='STI 014')

picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                   exclude='bads')

# Read epochs (train will be done only between 1 and 2s)
# Testing will be done with a running classifier
epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                baseline=None, preload=True)
epochs_train = epochs.copy().crop(tmin=1.5, tmax=3.5)
labels = epochs.events[:, -1] #- 2

###############################################################################
# Classification with linear discrimant analysis

# Define a monte-carlo cross-validation generator (reduce variance):
scores = []
epochs_data = epochs.get_data()
epochs_data_train = epochs_train.get_data()
cv = ShuffleSplit(10, test_size=0.2)
cv_split = cv.split(epochs_data_train)

# Assemble a classifier
lda = LinearDiscriminantAnalysis(n_components = 1)
csp = CSP(n_components=3, reg=None, log=True, norm_trace=False)


# Use scikit-learn Pipeline with cross_val_score function
clf = Pipeline([('CSP', csp), ('LDA', lda)])
scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1)

# Printing the results
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1. - class_balance)
print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
                                                          class_balance))

# plot CSP patterns estimated on full data for visualization
# csp.fit_transform(epochs_data, labels)
#
# layout = read_layout('EEG1005')
# csp.plot_patterns(epochs.info, layout=layout, ch_type='eeg',
#                   units='Patterns (AU)', size=1.5)

###############################################################################
# Look at performance over time

sfreq = raw.info['sfreq']
w_length = int(sfreq * 0.5)   # running classifier: window length
w_step = int(sfreq * 0.1)  # running classifier: window step size
w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step)

scores_windows = []

for train_idx, test_idx in cv_split:
    y_train, y_test = labels[train_idx], labels[test_idx]

    X_train = csp.fit_transform(epochs_data_train[train_idx], y_train)
    # X_test = csp.transform(epochs_data_train[test_idx])

    # fit classifier
    lda.fit(X_train, y_train)

    # running classifier: test classifier on sliding window
    score_this_window = []
    for n in w_start:
        X_test = csp.transform(epochs_data[test_idx][:, :, n:(n + w_length)])
        score_this_window.append(lda.score(X_test, y_test))
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
