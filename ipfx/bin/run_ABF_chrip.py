import pandas as pd
import numpy as np
import logging
import pyabf
from ipfx.sweep import Sweep,SweepSet
import ipfx.chirp as chirp
import ipfx.abf_dataset
import ipfx.feature_vectors as fv
import ipfx.time_series_utils as tsu
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack
import scipy.signal as signal
import tkinter as tk
from tkinter import filedialog
import os

root = tk.Tk()
root.withdraw()
files = filedialog.askdirectory(
                                   title='Select dir File'
                                   )
root_fold = files

#abf_chrip = pyabf.ATF('h:\\Sam\\Monkey\\Chirp Proto\\20mv Chirp.atf')
#abf_set = abf_dataset.ABFDataSet(abf_file=abf)
def abf_chirp(abf):
    t = abf.sweepX
    v = t
    abf_chrip = pyabf.ATF('H:\\Sam\\Protocol\\Monkey\\Chirp Proto\\Old Chirp\\20mv Chirp.atf')
    i = abf_chrip.sweepY[:-1]
    for x in range(0,abf.sweepCount):
        abf.setSweep(x)
        v = np.vstack((v,abf.sweepY))
        i = np.vstack((i,abf_chrip.sweepY[:-1]))
    v = v[1:]
    i = i[1:]
    t = abf.sweepX

    def chirp_amp_phase(v,i, t, start=0.3, end=19.68, down_rate=20000.0,
            min_freq=0.1, max_freq=17.5):
        """ Calculate amplitude and phase of chirp responses

        Parameters
        ----------
        sweep_set: SweepSet
            Set of chirp sweeps
        start: float (optional, default 0.6)
            Start of chirp stimulus in seconds
        end: float (optional, default 20.6)
            End of chirp stimulus in seconds
        down_rate: int (optional, default 2000)
            Sampling rate for downsampling before FFT
        min_freq: float (optional, default 0.2)
            Minimum frequency for output to contain
        max_freq: float (optional, default 40)
            Maximum frequency for output to contain

        Returns
        -------
        amplitude: array
            Aka resistance
        phase: array
            Aka reactance
        freq: array
            Frequencies for amplitude and phase results
        """
        v_list = v
        i_list = i
   


        avg_v = np.vstack(v_list).mean(axis=0)
        avg_i = np.vstack(i_list).mean(axis=0)

        plt.plot(t, avg_v)
        plt.plot(t, avg_i)
        
        current_rate = np.rint(1 / (t[1] - t[0]))
        if current_rate > down_rate:
            width = int(current_rate / down_rate)
            ds_v = ds_v = fv._subsample_average(avg_v, width)
            ds_i = fv._subsample_average(avg_i, width)
            ds_t = t[::width]
        else:
            ds_v = avg_v
            ds_i = avg_i
            ds_t = t

        start_index = tsu.find_time_index(ds_t, start)
        end_index = tsu.find_time_index(ds_t, end)

        N = len(ds_v[start_index:end_index])
        T = ds_t[1] - ds_t[0]
        xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

        v_fft = fftpack.fft(ds_v[start_index:end_index])
        i_fft = fftpack.fft(ds_i[start_index:end_index])
        Z = v_fft / i_fft
        R = np.real(Z)
        X = np.imag(Z)
        resistance = np.abs(Z)[0:N//2]
        reactance = np.arctan(X / R)[0:N//2]

        low_ind = tsu.find_time_index(xf, min_freq)
        high_ind = tsu.find_time_index(xf, max_freq)
        v2 = R[0:N//2]
        return resistance[low_ind:high_ind], reactance[low_ind:high_ind], xf[low_ind:high_ind], v2[low_ind:high_ind]

    resist, react, _, v2 = chirp_amp_phase(v,i,t)
    return resist, react, _, v2

len_f = 400

full = np.full(len_f , np.nan)
for root,dir,fileList in os.walk(files):
 for filename in fileList:
    if filename.endswith(".abf"):
        file_path = os.path.join(root,filename)
        abf = pyabf.ABF(file_path)
        if abf.dataRate > 10000:
            print(abf.abfID + ' loaded')
            abf_name = np.vstack([abf.abfID,abf.abfID, abf.abfID, abf.abfID, abf.abfID])
            abf_label = np.vstack(['resist','react', 'freq','detrend resist', 'v2'])
            abf_feat = abf_chirp(abf)
            abf_feat = np.vstack((abf_feat, signal.detrend(abf_feat[0])))
            abf_ar = np.hstack((abf_name, abf_label, abf_feat))
            abf_ar = np.hstack((abf_ar, np.vstack([np.full(len_f  - abf_ar.shape[1], np.nan), np.full(len_f  -abf_ar.shape[1], np.nan), np.full(len_f  -abf_ar.shape[1], np.nan), np.full(len_f  -abf_ar.shape[1], np.nan), np.full(len_f  -abf_ar.shape[1], np.nan)])))
            full = np.vstack((full, abf_ar))

np.savetxt('H:\Sam\CHIRP.csv', full, delimiter=",", fmt='%s')
plt.show()
