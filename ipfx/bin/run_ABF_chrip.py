
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

def moving_avg(ar, window):
    ar_fix = np.hstack((ar, np.full(1000 - ar.shape[0], np.nan)))
    size = int(ar_fix.shape[0]/window)

    ar_change = np.vstack(np.split(ar_fix,size))
    running_mean = np.nanmean(ar_change, axis=1, keepdims=True)
    repeat = np.repeat(running_mean,window)
    return repeat


def moving_avg2(ar, window):
    series_ar = pd.Series(data=ar)
    running_mean = series_ar.rolling(window).mean().to_numpy()
    series_ar = pd.Series(data=np.flip(ar))
    running_mean_start = series_ar.rolling(window).mean().to_numpy()
    replace = int(window-1)
    running_mean[:replace] = running_mean_start[-replace:]
    return running_mean


def find_peak(x, y, freq_cut=0):
    y = y[x>freq_cut]
    x = x[x>freq_cut]
    ds_mean = np.nanmean(y)
    width_peak = 101
    peaks = np.array([[]])
    min_peaks = [np.full(999,9), 0]
    while width_peak > 0:
     
        peaks = signal.find_peaks(y, height=ds_mean, width=width_peak)
        if len(peaks[0]) < len(min_peaks[0]):
            min_peaks = peaks
        if len(peaks[0]) <= 2 and len(peaks[0]) >= 1:
            min_peaks = peaks
            break;
        width_peak -=1
    min_peaks[1]['x-heights'] = x[min_peaks[0]]
    return [min_peaks[1], width_peak]
    
#abf_chrip = pyabf.ATF('h:\\Sam\\Monkey\\Chirp Proto\\20mv Chirp.atf')
#abf_set = abf_dataset.ABFDataSet(abf_file=abf)
def abf_chirp(abf):
    t = abf.sweepX
    v = t
    abf_chrip = pyabf.ATF('H:\\Sam\\Protocol\\Monkey\\Chirp Proto\\0_20hz_in50s_vs.atf')
    i = abf_chrip.sweepY[:]
    for x in range(0,abf.sweepCount):
        abf.setSweep(x)
        v = np.vstack((v,abf.sweepY))
        i = np.vstack((i,abf_chrip.sweepY[:]))
    v = v[1:]
    i = i[1:]
    t = abf.sweepX

    def chirp_amp_phase(v,i, t, start=0.78089, end=49.21, down_rate=20000.0,
            min_freq=0.1, max_freq=19.5):
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

        #plt.plot(t, avg_v)
        #
        #plt.plot(t, avg_i)
        
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

len_f = 1000
peaks = pd.DataFrame()
full = np.full(len_f , np.nan)
for root,dir,fileList in os.walk(files):
 for filename in fileList:
    if filename.endswith(".abf"):
        file_path = os.path.join(root,filename)
        abf = pyabf.ABF(file_path)
        if abf.dataRate > 10000:
            print(abf.abfID + ' loaded')
            abf_name = np.vstack([abf.abfID,abf.abfID, abf.abfID, abf.abfID, abf.abfID])
            abf_label = np.vstack(['resist','react', 'freq','Z', 'resist running avg'])
            abf_feat = abf_chirp(abf)
            running_mean =  moving_avg2(abf_feat[0], 5)#[:abf_feat[0].shape[0]]

            tpeaks = find_peak(abf_feat[2], running_mean)
            temp = pd.DataFrame().from_dict(tpeaks[0])
            temp['id'] = np.full(temp.index.values.shape[0], abf.abfID)
            temp['width'] = np.full(temp.index.values.shape[0], tpeaks[1])
            peaks = peaks.append(temp)
            abf_feat = np.vstack((abf_feat, running_mean))
            abf_ar = np.hstack((abf_name, abf_label, abf_feat))
            abf_ar = np.hstack((abf_ar, np.vstack([np.full(len_f  - abf_ar.shape[1], np.nan), np.full(len_f  -abf_ar.shape[1], np.nan), np.full(len_f  -abf_ar.shape[1], np.nan), np.full(len_f  -abf_ar.shape[1], np.nan), np.full(len_f  -abf_ar.shape[1], np.nan)])))
            full = np.vstack((full, abf_ar))

np.savetxt('H:\Sam\CHIRP.csv', full, delimiter=",", fmt='%s')
peaks.to_csv("H:\Sam\CHIRP_resist_peaks.csv")
plt.show()
