from myClient import ScanClient
from classifier import myClassify
import threading
import numpy as np
import mne

class TransportProcess():
    def __init__(self,host,port):

        self.raw_eeg_of_single_trial = []
        self._process_thread = None

        self.nchan = 0
        self.info = None


        self.classify = myClassify() #todo

        self.scanClient = ScanClient(host,port)
        self.scanClient.register_receive_callback(self.setup_file_process)
        self.scanClient.register_receive_callback(self.push_block_raw_data)
        self.scanClient.start_receive_thread()


        #todo edf and ast file parse
        self.nchan = 0
        self.countOfBytes = 0

    def begin_record(self):
        self.scanClient.start_sending_data()
        self.raw_eeg_of_single_trial = []

    def single_trial_end(self):
        self.scanClient.stop_sending_data()
        #todo create a new thread for process eeg data
        self._process_thread = threading.Thread(target=self.raw_eeg_data_process)
        self._process_thread.start()

    def edf_head_process(self):
        #todo parse nchan etc.
        #edf头文件里有每个数据所占位数？
        pass

    def setup_file_process(self):
        self.nchan = 64
        sfreq = 1000
        mne.create_info()
        info = mne.create_info(
            ch_names=['MEG1', 'MEG2', 'EEG1', 'EEG2', 'EOG'],
            ch_types=['grad', 'grad', 'eeg', 'eeg', 'eog'],
            sfreq=sfreq
        )
        self.info = info
        #todo 补全

    # def _16_bit_data_process(self,buffer):
    #     pass
    #
    # def _32_bit_data_process(self,buffer):
    #     pass

    def raw_eeg_data_process(self, raw_buffer):
        single_trial_raw = mne.io.RawArray(self.raw_eeg_of_single_trial,self.info)


    def push_block_raw_data(self,raw_buffer):
        #todo reshape(-1,nchan) and  push to raw_eeg_of_single_trial
        if raw_buffer['head'][0][2] == 2:
            temp_buffer = np.frombuffer(raw_buffer['buffer'],'>i4').reshape(self.nchan,-1)
        elif raw_buffer['head'][0][2] == 1:
            temp_buffer = np.frombuffer(raw_buffer['buffer'], '>i2').reshape(self.nchan,-1)

        self.raw_eeg_of_single_trial.append(temp_buffer)


