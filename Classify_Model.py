from mne.decoding import FilterEstimator
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer,MaxAbsScaler
from sklearn.model_selection import ShuffleSplit, cross_val_score,KFold
from EEG_Data_Loader import EEG_Data_Loader

import numpy as np

class ClassifyModel(object):
    def __init__(self, t_winSize, t_stride, trainFilename):

        self.eeg_data_loader = EEG_Data_Loader()
        self.train_data,self.labels = self.eeg_data_loader.load_train_data(t_winSize, t_stride,trainFilename)
        print(self.train_data.shape,len(self.labels),self.labels)
        self.shuffle_data_label()
        print(self.labels)
        self.csp = None #CSP(n_components=5, reg=None, log=True, norm_trace=False)
        self.lda = LinearDiscriminantAnalysis()
        self.scaler_befor_lda = Normalizer()
        self.scaler_after_lda = MaxAbsScaler()
        self.clf = None #Pipeline([('CSP', self.csp), ('SCALER_BEFOR', self.scaler_befor_lda), ('LDA', self.lda)])
        # hyper_parameter
        self.csp_component = 5

    def train_model(self): #X = (nTrial,nChannel,nTimes)
        self.csp = CSP(n_components=self.csp_component, reg=None, log=True, norm_trace=False)
        self.clf = Pipeline([('CSP', self.csp), ('SCALER_BEFOR', self.scaler_befor_lda),
                             ('LDA', self.lda)]) #,('SCALER_AFTER',self.scaler_after_lda)
        self.clf.fit(self.train_data,self.labels)
        trans_result = self.clf.transform(self.train_data)
        self.scaler_after_lda.fit(trans_result)

    def transform(self,X):
        if self.clf is not None:
            trans_result = self.clf.transform(X)
            print('trans_result',trans_result)
            scaler_result = self.scaler_after_lda.transform(trans_result)
            print('scaler_result',scaler_result)
            return scaler_result
        else:
            return None

    def update_model(self):
        pass

    def cross_val_score(self):
        # cv = ShuffleSplit(10, test_size=0.2, random_state=12)
        cv = KFold(n_splits=5,shuffle=True)
        scores_list = []
        for c in range(3, 10):
            csp = CSP(n_components=c, reg=None, log=True, norm_trace=False)
            clf = Pipeline([('CSP', csp), ('SCALER_BEFOR', self.scaler_befor_lda),
                                 ('LDA', self.lda)])  # ('SCALER_AFTER',self.scaler_after_lda)
            scores = cross_val_score(clf, self.train_data, self.labels, cv=cv, n_jobs=1) #todo 交叉验证的内部机制如何保证稳定的准确率
            scores_list.append((c, round(float(np.mean(scores)), 3)))
            # Printing the results
        class_balance = np.mean(self.labels == self.labels[0])
        class_balance = max(class_balance, 1. - class_balance)
        scores_list.sort(key=lambda x: x[1], reverse=True)
        self.csp_component = scores_list[0][0]
        return scores_list,class_balance

    def shuffle_data_label(self):
        state = np.random.get_state()
        np.random.shuffle(self.train_data)
        np.random.set_state(state)
        np.random.shuffle(self.labels)


if __name__ == "__main__":
    clf_model = ClassifyModel(t_winSize=3,t_stride=3,trainFilename="D:/temp/EEG_DATA/EEGData2/静息睁眼.mat")
    #D:\temp\EEG_DATA\EEGData2\真实左右肘动_30_手动控制.mat #D:/temp/EEG_DATA/EEGDATA4/mxl左右手想象20次四秒5bd.mat
    #D:/temp/EEG_DATA/EEGData2/静息睁眼.mat #D:/temp/EEG_DATA/EEGDATA4/mxl左右手想象20次四秒3.mat
    scores_list, class_balance = clf_model.cross_val_score()

    print(scores_list,class_balance)

    # clf_model.train_model()

