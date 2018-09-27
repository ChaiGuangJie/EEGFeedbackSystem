from mne.decoding import FilterEstimator
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from EEG_Data_Loader import EEG_Data_Loader

class ClassifyModel(object):
    def __init__(self,trainFilename):

        self.eeg_data_loader = EEG_Data_Loader()
        self.train_data,self.labels = self.eeg_data_loader.load_train_data(trainFilename)
        self.csp = CSP(n_components=5, reg=None, log=True, norm_trace=False)
        self.lda = LinearDiscriminantAnalysis()

    def train_model(self):
        pass #todo return fetures #[(x,y,label),...]

    def transform(self):
        pass #todo return (x,y,label)

    def update_model(self):
        pass

#
# class csp_lda_clf(ClassifyModel):
#     def __init__(self):
#         super().__init__()
#         pass

