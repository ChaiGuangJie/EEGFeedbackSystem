import socket,time,threading
import numpy as np
import mne
from mne.utils import verbose

basicInfoType = np.dtype([('dwSize',np.uint32),('nEegChan',np.int32),('nEvtChan',np.int),('nBlockPnts',np.int),('nRate',np.int),('nDataSize',np.int)])
headType = np.dtype([('IDString','>S4'),('Code','>u2'),('Request','>u2'),('BodySize','>u4')])

def _buffer_recv_worker(rt_client, nchan):
    """Worker thread that constantly receives buffers."""
    try:
        for raw_buffer in rt_client.raw_buffers(nchan):
            rt_client._push_raw_buffer(raw_buffer)
    except RuntimeError as err:
        # something is wrong, the server stopped (or something)
        rt_client._recv_thread = None
        print('Buffer receive thread stopped: %s' % err)

def _recv_head_raw(sock):
    """Read a head and the associated data from a socket.

    Parameters
    ----------
    sock : socket.socket
        The socket from which to read the tag.

    Returns
    -------
    head : array of head
        The head.
    buff : str
        The raw data of the tag (including header).
    """
    s = sock.recv(12)
    if len(s) != 12:
        raise RuntimeError('Not enough head bytes received, something is wrong. '
                           'Make sure the server is running.')
    head = np.frombuffer(s, headType)
    n_received = 0
    rec_buff = []  # 只包含data body,不含header
    while n_received < int(head[0]['BodySize']):
        n_buffer = min(4096, int(head[0]['BodySize']) - n_received)
        this_buffer = sock.recv(n_buffer)
        rec_buff.append(this_buffer)
        n_received += len(this_buffer)

    if n_received != int(head[0]['BodySize']):
        raise RuntimeError('Not enough bytes received, something is wrong. '
                           'Make sure the server is running.')

    buff = b''.join(rec_buff)

    return head, buff


class ScanClient(object):

    @verbose
    def __init__(self,host,port,timeout=10,verbose=None):
        self._host = host,
        self._port = port,
        self._timeout = timeout
        self.commandDict = {
            'Request_for_Version': self._format_head('CTRL',1,1,0),
            'Closing_Up_Connection':self._format_head('CTRL',1,2,0),
            'Start_Acquisition':self._format_head('CTRL',2,1,0),
            'Stop_Acquisition': self._format_head('CTRL', 2, 2, 0),
            'Start_Impedance': self._format_head('CTRL', 2, 3, 0),
            'Change_Setup': self._format_head('CTRL', 2, 4, 0),
            'DC_Correction': self._format_head('CTRL', 2, 5, 0),
            'Request_for_EDF_Header': self._format_head('CTRL', 3, 1, 0),
            'Request_for_AST_Setup_File': self._format_head('CTRL', 3, 2, 0),
            'Request_to_Start_Sending_Data': self._format_head('CTRL', 3, 3, 0),
            'Request_to_Stop_Sending_Data': self._format_head('CTRL', 3, 4, 0),
            'Request_for_basic_info':self._format_head('CTRL', 3, 5, 0),
            'Neuroscan_16bit_Raw_Data':self._format_head('Data',2,1,0),
            'Neuroscan_32bit_Raw_Data': self._format_head('Data', 2, 2, 0)
        }

        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(timeout)
            self._sock.connect((host, port))
            self._sock.setblocking(True)
        except Exception:
            raise RuntimeError('Setting up command connection (host: %s '
                               'port: %d) failed. Make sure server '
                               'is running. ' % (host, port))

        self._recv_callbacks = list()
        self.TransPortActivate = False
        self._recv_thread = None

        self.verbose = verbose

        self.current_trigger = None
        self.trigger_lock = threading.Lock()

    def get_measurement_info(self):
        """Get the measurement information.

        Returns
        -------
        info : dict
            The measurement information.
        """
        #必须有个channel是trigger 且在ch_names里标识
        cmd =self.commandDict['Request_for_basic_info']
        self._send_command(cmd) #todo 先获取信息，再获取数据
        head,buffer = _recv_head_raw(self._sock)

        #todo 判断是否是返回的basicInfo
        basicInfo = np.array(buffer, basicInfoType)
        nEegChan = basicInfo['nEegChan'] # EEG通道数 67
        nEvtChan = basicInfo['nEvtChan'] #event 通道个数 1
        nBlockPnts = basicInfo['nBlockPnts'] #block点数 40
        nRate = basicInfo['nRate'] #采样率 1000
        nDataSize = basicInfo['nDataSize'] #一个数据占用字节数 4

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

        return mne.create_info(ch_names, nRate, 'eeg') #todo 可以通过读取ast文件详细赋值或手动输入 #nEegChan + nEvtChan


    def start_receive_thread(self,nchan):
        """Start the receive thread.

        If the measurement has not been started, it will also be started.

        Parameters
        ----------
        nchan : int
            The number of channels in the data.
        """
        if self._recv_thread is None:
            #self.start_sending_data()

            self._recv_thread = threading.Thread(target=_buffer_recv_worker,args=(self, nchan))
            self._recv_thread.start()

    # todo 客户端单方面无法只退出接收线程，只能主动请求断开链接后，接收线程自动退出
    def stop_receive_thread(self, stop_measurement=True):
        """Stop the receive thread.

        Parameters
        ----------
        stop_measurement : bool
            Also stop the measurement.
        """
        if stop_measurement:
            self.stop_measurement()

        if self._recv_thread is not None:
            self._recv_thread.join() #todo 读取线程如何强制结束？
            self._recv_thread = None

    def stop_recv_and_disconnect(self):

        self.stop_measurement()
        if self._recv_thread is not None:
            self._recv_thread.join()
            self._recv_thread = None

        self._sock.close()

    def stop_measurement(self):
        """Stop the measurement."""
        self.stop_sending_data()
        time.sleep(0.1)
        self._close_connect()

    def raw_buffers(self,nchan):
        """Return an iterator over raw buffers.

        Parameters
        ----------
        nchan : int
            The number of channels (info['nchan']).

        Returns
        -------
        raw_buffer : generator
            Generator for iteration over raw buffers.
        """
        while True:
            raw_buffer = self.read_raw_buffer(nchan)
            if raw_buffer is not None: #TODO 何时为None
                yield raw_buffer
            else:
                break

    def read_raw_buffer(self,nchan):
        try:
            head,buffer = _recv_head_raw(self._sock)
        except RuntimeError as err:
            print(err)
            return None

        if head[0][2] == 2:
            buffer_array = np.frombuffer(buffer,'<i4').reshape(-1, nchan).T
        elif head[0][2] == 1:
            buffer_array = np.frombuffer(buffer, '<i2').reshape(-1, nchan).T
        else:
            buffer_array =None
        #todo 手动打标签
        if buffer_array is not None:
            buffer_array = buffer_array.copy()
            buffer_array = buffer_array * 0.00015
            buffer_array[-1,:] = 0
            if self.current_trigger is not None:
                buffer_array[-1,0:3] = self.current_trigger
                # print('buffer_array',buffer_array[-1,0])
                self.set_event_trigger(None)
        return buffer_array
        #必须返回shape=(nchan, n_times)格式的数据

    def set_event_trigger(self,trigger):
        self.trigger_lock.acquire()
        self.current_trigger = trigger
        # print('current_trigger',self.current_trigger)
        self.trigger_lock.release()

    def register_receive_callback(self, callback):
        """Register a raw buffer receive callback.

        Parameters
        ----------
        callback : callable
            The callback. The raw buffer is passed as the first parameter
            to callback.
        """
        if callback not in self._recv_callbacks:
            self._recv_callbacks.append(callback)

    def unregister_receive_callback(self, callback):
        """Unregister a raw buffer receive callback.

        Parameters
        ----------
        callback : function
            The callback to unregister.
        """
        if callback in self._recv_callbacks:
            self._recv_callbacks.remove(callback)

    def _push_raw_buffer(self, raw_buffer):
        """Push raw buffer to clients using callbacks."""
        for callback in self._recv_callbacks:
            callback(raw_buffer)

    def _format_head(self, IDString, Code, Request, BodySize):
        return np.array([(IDString, Code, Request, BodySize)], dtype=headType)  # .tobytes()

    def _send_command(self, command):
        self._sock.sendall(command.tobytes())

    def _close_connect(self):
        # self._send_command(self.commandDict['Request_to_Stop_Sending_Data'])
        # time.sleep(0.1)  # 是否需要等服务器回应？
        self._send_command(self.commandDict['Closing_Up_Connection'])


    def request_EDF_header(self):
        self._send_command(self.commandDict['Request_for_EDF_Header'])

    def start_sending_data(self):
        self._send_command(self.commandDict['Request_to_Start_Sending_Data'])

    def stop_sending_data(self):
        self._send_command(self.commandDict['Request_to_Stop_Sending_Data'])




if __name__ == '__main__':
    c = ScanClient('10.0.180.151',4000)
    # c = ScanClient('127.0.0.1',5555,5)
    info = c.get_measurement_info()

    c.start_receive_thread(info['nchan'])
    def show_rect(buffer):
        # print([b*0.00015 for b in buffer])
        print(buffer)
        print(len(buffer))
        # print(buffer['head'][0][2]==2)
    c.register_receive_callback(show_rect)
    time.sleep(1)
    test = np.array([('CTRL',3,3,0)],dtype=headType)#start sending data
    # test = np.array([('CTRL',3,5,0)],dtype=headType)#basic info

    # test = b'\x00\x01\x00\x02\x00\x00\x00\x00'
    time.sleep(5)
    c._send_command(test)
    # time.sleep(1)
    # head,buff = c._recv_head_raw()
    # print(head)
    # print(buff)
    # print(test.dtype)
    # print(test)

    wait = input('input something to end:')
    c.stop_recv_and_connect()
    # try:
    #     # c.stop_receive_thread(stop_measurement=True)
    #     # c._send_command(c.commandDict['Request_to_Stop_Sending_Data'])
    #     # time.sleep(3)
    #     # print('stop recvd!')
    #     # c._close_connect()
    #     # time.sleep(1)
    #     # print('stop connect!')
    #     # c._sock.close()
    # except ConnectionAbortedError:
    #     print('服务器已断开链接！')
    print('end')

