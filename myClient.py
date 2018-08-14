import socket,time,threading
import numpy as np
from mne.realtime.client import _buffer_recv_worker

headType = np.dtype([('IDString','>S4'),('Code','>u2'),('Request','>u2'),('BodySize','>u4')])

# class CommandHeader():
#     def __init__(self,IDString,Code,Request,BodySize):
#         self.headCommand = np.array([(IDString,Code,Request,BodySize)],dtype = headType)

class ScanClient():
    def __init__(self,host,port,timeout=10):
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
        self.nchan = 67

    def start_receive_thread(self):
        """Start the receive thread.

        If the measurement has not been started, it will also be started.

        Parameters
        ----------
        nchan : int
            The number of channels in the data.
        """
        if self._recv_thread is None:
            #self.start_sending_data()

            self._recv_thread = threading.Thread(target=self._buffer_recv_worker)
            self._recv_thread.start()

    def _buffer_recv_worker(self):
        """Worker thread that constantly receives buffers."""
        try:
            for raw_buffer in self.raw_buffers():
                self._push_raw_buffer(raw_buffer)
        except RuntimeError as err:
            # something is wrong, the server stopped (or something)
            self._recv_thread = None
            self.stop_sending_data() #todo 是否需要？
            print('Buffer receive thread stopped: %s' % err)

    def raw_buffers(self):
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
            raw_buffer = self.read_raw_buffer()
            if raw_buffer is not None: #TODO 何时为None
                yield raw_buffer
            else:
                break

    def read_raw_buffer(self):
        try:
            s = self._sock.recv(12) #12 bytes in header
        except ConnectionAbortedError:
            return None
        if len(s) != 12:
            raise RuntimeError('Not enough bytes received, something is wrong. '
                               'Make sure the server is running.')
        head = np.frombuffer(s, headType)
        n_received = 0
        rec_buff = [] #只包含data body,不含header
        while n_received < int(head[0]['BodySize']):
            n_buffer = min(4096, int(head[0]['BodySize']) - n_received)
            this_buffer = self._sock.recv(n_buffer)
            rec_buff.append(this_buffer)
            n_received += len(this_buffer)

        if n_received != int(head[0]['BodySize']):
            raise RuntimeError('Not enough bytes received, something is wrong. '
                               'Make sure the mne_rt_server is running.')
        buffer = b''.join(rec_buff)
        # buffer = np.frombuffer(b''.join(rec_buff), '<i4')
        # buffer = buffer.reshape(-1, self.nchan).T
        return buffer  #{'head' : head, 'buffer' : buffer}
        #todo 必须返回shape=(nchan, n_times)格式的数据 其中必须有个channel是trigger 且在ch_names里标识

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
        self._send_command(self.commandDict['Request_to_Stop_Sending_Data'])
        time.sleep(0.1)  # todo 是否需要等服务器回应？
        self._send_command(self.commandDict['Closing_Up_Connection'])
        time.sleep(0.1)  # todo 是否需要等服务器回应？
        self._sock.close()

    def request_EDF_header(self):
        self._send_command(self.commandDict['Request_for_EDF_Header'])

    def start_sending_data(self):
        self._send_command(self.commandDict['Request_to_Start_Sending_Data'])

    def stop_sending_data(self):
        self._send_command(self.commandDict['Request_to_Stop_Sending_Data'])




if __name__ == '__main__':
    c = ScanClient('10.0.180.151',4000,5)
    # c = ScanClient('127.0.0.1',5555,5)

    c.start_receive_thread()
    def show_rect(buffer):
        # print([b*0.00015 for b in buffer])
        print(buffer)
        print(len(buffer))
        # print(buffer['head'][0][2]==2)
    c.register_receive_callback(show_rect)
    time.sleep(1)
    # test = np.array([('CTRL',3,3,0)],dtype=headType)
    test = np.array([('CTRL',3,5,0)],dtype=headType)#basic info

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
    try:
        c._close_connect()
    except ConnectionAbortedError:
        print('服务器已断开链接！')

