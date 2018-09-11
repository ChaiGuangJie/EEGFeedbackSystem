from mne.realtime import RtEpochs
import numpy as np
from mne.event import _find_events


def _epochs_next(self, return_event_id=False):
    pass

def _process_raw_buffer(self, raw_buffer):
    """Process raw buffer (callback from RtClient).

    Note: Do not print log messages during regular use. It will be printed
    asynchronously which is annoying when working in an interactive shell.

    Parameters
    ----------
    raw_buffer : array of float, shape=(nchan, n_times)
        The raw buffer.
    """
    sfreq = self.info['sfreq']
    n_samp = len(self._raw_times)

    # relative start and stop positions in samples
    tmin_samp = int(round(sfreq * self.tmin))
    tmax_samp = tmin_samp + n_samp

    last_samp = self._first_samp + raw_buffer.shape[1] - 1

    # apply calibration without inplace modification
    raw_buffer = self._cals * raw_buffer

    # detect events
    data = np.abs(raw_buffer[self._stim_picks]).astype(np.int)
    # if there is a previous buffer check the last samples from it too
    if self._last_buffer is not None:
        prev_data = self._last_buffer[self._stim_picks,
                    -raw_buffer.shape[1]:].astype(np.int)
        data = np.concatenate((prev_data, data), axis=1)
        data = np.atleast_2d(data)
        buff_events = _find_events(data,
                                   self._first_samp - raw_buffer.shape[1],
                                   **self._find_events_kwargs)
        # print('buff_events', len(buff_events))

    else:
        data = np.atleast_2d(data)
        buff_events = _find_events(data, self._first_samp,
                                   **self._find_events_kwargs)
    events = self._event_backlog

    # remove events before the last epoch processed
    min_event_samp = self._first_samp - \
                     int(self._find_events_kwargs['min_samples'])
    if len(self._event_backlog) > 0:
        backlog_samps = np.array(self._event_backlog)[:, 0]
        min_event_samp = backlog_samps[-1] + 1

    if buff_events.shape[0] > 0:
        valid_events_idx = buff_events[:, 0] >= min_event_samp
        buff_events = buff_events[valid_events_idx]

    # add events from this buffer to the list of events
    # processed so far
    for event_id in self.event_id.values():
        idx = np.where(buff_events[:, -1] == event_id)[0]
        events.extend(zip(list(buff_events[idx, 0]),
                          list(buff_events[idx, -1])))

    events.sort()

    event_backlog = list()
    for event_samp, event_id in events:
        epoch = None
        if (event_samp + tmin_samp >= self._first_samp and
                event_samp + tmax_samp <= last_samp):
            # easy case: whole epoch is in this buffer
            start = event_samp + tmin_samp - self._first_samp
            stop = event_samp + tmax_samp - self._first_samp
            epoch = raw_buffer[:, start:stop]
        elif (event_samp + tmin_samp < self._first_samp and
              event_samp + tmax_samp <= last_samp):
            # have to use some samples from previous buffer
            if self._last_buffer is None:
                continue
            n_last = self._first_samp - (event_samp + tmin_samp)
            n_this = n_samp - n_last
            epoch = np.c_[self._last_buffer[:, -n_last:],
                          raw_buffer[:, :n_this]]
        elif event_samp + tmax_samp > last_samp:
            # we need samples from the future
            # we will process this epoch with the next buffer
            event_backlog.append((event_samp, event_id))
        else:
            raise RuntimeError('Unhandled case..')

        if epoch is not None:
            self._append_epoch_to_queue(epoch, event_samp, event_id)
    # print(len(self._epoch_queue))
    # set things up for processing of next buffer
    self._event_backlog = event_backlog
    n_buffer = raw_buffer.shape[1]
    if self._last_buffer is None:
        self._last_buffer = raw_buffer
        self._first_samp = last_samp + 1
    elif self._last_buffer.shape[1] <= n_samp + n_buffer:
        self._last_buffer = np.c_[self._last_buffer, raw_buffer]
    else:
        # do not increase size of _last_buffer any further
        self._first_samp = self._first_samp + n_buffer
        self._last_buffer[:, :-n_buffer] = self._last_buffer[:, n_buffer:]
        self._last_buffer[:, -n_buffer:] = raw_buffer


'''
class MyRtEpoch(RtEpochs):
    def __init__(self, client, event_id, tmin, tmax, stim_channel='STI 014',
                 sleep_time=0.1, baseline=(None, 0), picks=None,
                 reject=None, flat=None, proj=True,
                 decim=1, reject_tmin=None, reject_tmax=None, detrend=None,
                 isi_max=2., find_events=None, verbose=None):  # noqa: D102
        super(MyRtEpoch,self).__init__(client = client, event_id =event_id, tmin=tmin, tmax=tmax,
                                       stim_channel=stim_channel,sleep_time=sleep_time, baseline=baseline,
                                       picks=picks,reject=reject, flat=flat, proj=proj,decim=decim,
                                       reject_tmin=reject_tmin, reject_tmax=reject_tmax, detrend=detrend,
                                       isi_max=isi_max, find_events=find_events, verbose=verbose)

    def _process_raw_buffer(self, raw_buffer):
        """Process raw buffer (callback from RtClient).

        Note: Do not print log messages during regular use. It will be printed
        asynchronously which is annoying when working in an interactive shell.

        Parameters
        ----------
        raw_buffer : array of float, shape=(nchan, n_times)
            The raw buffer.
        """
        sfreq = self.info['sfreq']
        n_samp = len(self._raw_times)

        # relative start and stop positions in samples
        tmin_samp = int(round(sfreq * self.tmin))
        tmax_samp = tmin_samp + n_samp

        last_samp = self._first_samp + raw_buffer.shape[1] - 1

        # apply calibration without inplace modification
        raw_buffer = self._cals * raw_buffer

        # detect events
        data = np.abs(raw_buffer[self._stim_picks]).astype(np.int)
        # if there is a previous buffer check the last samples from it too
        if self._last_buffer is not None:
            prev_data = self._last_buffer[self._stim_picks,
                                          -raw_buffer.shape[1]:].astype(np.int)
            data = np.concatenate((prev_data, data), axis=1)
            data = np.atleast_2d(data)
            buff_events = _find_events(data,
                                       self._first_samp - raw_buffer.shape[1],
                                       **self._find_events_kwargs)
            # print('buff_events', len(buff_events))

        else:
            data = np.atleast_2d(data)
            buff_events = _find_events(data, self._first_samp,
                                       **self._find_events_kwargs)
        events = self._event_backlog

        # remove events before the last epoch processed
        min_event_samp = self._first_samp - \
            int(self._find_events_kwargs['min_samples'])
        if len(self._event_backlog) > 0:
            backlog_samps = np.array(self._event_backlog)[:, 0]
            min_event_samp = backlog_samps[-1] + 1

        if buff_events.shape[0] > 0:
            valid_events_idx = buff_events[:, 0] >= min_event_samp
            buff_events = buff_events[valid_events_idx]

        # add events from this buffer to the list of events
        # processed so far
        for event_id in self.event_id.values():
            idx = np.where(buff_events[:, -1] == event_id)[0]
            events.extend(zip(list(buff_events[idx, 0]),
                              list(buff_events[idx, -1])))

        events.sort()

        event_backlog = list()
        for event_samp, event_id in events:
            epoch = None
            if (event_samp + tmin_samp >= self._first_samp and
                    event_samp + tmax_samp <= last_samp):
                # easy case: whole epoch is in this buffer
                start = event_samp + tmin_samp - self._first_samp
                stop = event_samp + tmax_samp - self._first_samp
                epoch = raw_buffer[:, start:stop]
            elif (event_samp + tmin_samp < self._first_samp and
                    event_samp + tmax_samp <= last_samp):
                # have to use some samples from previous buffer
                if self._last_buffer is None:
                    continue
                n_last = self._first_samp - (event_samp + tmin_samp)
                n_this = n_samp - n_last
                epoch = np.c_[self._last_buffer[:, -n_last:],
                              raw_buffer[:, :n_this]]
            elif event_samp + tmax_samp > last_samp:
                # we need samples from the future
                # we will process this epoch with the next buffer
                event_backlog.append((event_samp, event_id))
            else:
                raise RuntimeError('Unhandled case..')

            if epoch is not None:
                self._append_epoch_to_queue(epoch, event_samp, event_id)
        # print(len(self._epoch_queue))
        # set things up for processing of next buffer
        self._event_backlog = event_backlog
        n_buffer = raw_buffer.shape[1]
        if self._last_buffer is None:
            self._last_buffer = raw_buffer
            self._first_samp = last_samp + 1
        elif self._last_buffer.shape[1] <= n_samp + n_buffer:
            self._last_buffer = np.c_[self._last_buffer, raw_buffer]
        else:
            # do not increase size of _last_buffer any further
            self._first_samp = self._first_samp + n_buffer
            self._last_buffer[:, :-n_buffer] = self._last_buffer[:, n_buffer:]
            self._last_buffer[:, -n_buffer:] = raw_buffer
'''