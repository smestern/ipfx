
import pandas as pd
import numpy as np
import logging
import pyabf
from ipfx import feature_vectors as fv
import ipfx.time_series_utils as tsu
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack
from scipy.interpolate import interp1d
import scipy.signal as signal
import tkinter as tk
from tkinter import filedialog
import os
import glob
dir_script = os.path.dirname(os.path.abspath(__file__))
atf_files = glob.glob(dir_script+"\\*.atf")
_20_in_50 = dir_script + "\\0_20hz_in50s_vs.atf"
_20_in_30 = dir_script + "\\0_20hz_in30s.atf"
_50_in_50 = dir_script + "\\0_50hz_in50s_vs.atf"



def _subsample_average(x, width):
    """Downsamples x by averaging `width` points"""

    avg = np.nanmean(x.reshape(-1, width), axis=1)
    return avg



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
            break
        width_peak -=1
    min_peaks[1]['x-heights'] = x[min_peaks[0]]
    # plt.clf()
    # plt.plot(x, y)
    # plt.plot(x[peaks[0]], y[peaks[0]], "x", label='found_peak')
    # plt.vlines(x=x[peaks[0]], ymin=y[peaks[0]] - peaks[1]["prominences"],
    #          ymax = y[peaks[0]], color = "C1", label='prominance')
    # plt.hlines(y=peaks[1]["width_heights"], xmin=x[int(peaks[1]["left_ips"])],
    #          xmax=x[int(peaks[1]["right_ips"])], color = "C1", label='IPS width / half maximal')
    # plt.legend()
    # plt.show()
    return [min_peaks[1], width_peak]
    

def analyze_abf_chirp(abf, stimuli_abf, average='input', min_freq=0.1, max_freq=10):
    t = abf.sweepX
    v = t
    i = stimuli_abf.sweepY[:]
    if average=='input':
        for x in range(0,abf.sweepCount):
            abf.setSweep(x)
            v = np.vstack((v,abf.sweepY))
            i = np.vstack((i,stimuli_abf.sweepY[:]))
        v = v[1:]
        i = i[1:]
        t = abf.sweepX
        v, i, t = preprocess_data(v, i, t)
        resist, react, z = chirp_amp_phase(v,i,t, min_freq=min_freq, max_freq=max_freq)
    elif average=='output':
        resistance = []
        reactance = []
        for x in range(0,abf.sweepCount):
            abf.setSweep(x)
            v = abf.sweepY
            i = stimuli_abf.sweepY[:]
            v, i, t = preprocess_data(v, i, t, average=False)
            temp_resist, temp_react, temp_z = chirp_amp_phase(v,i,t, min_freq=min_freq, max_freq=max_freq)
            resistance.append(temp_resist)
            reactance.append(temp_react)
        resist = np.nanmean(np.vstack(resistance), axis=0)
        react = np.nanmean(np.vstack(reactance), axis=0)
        z = temp_z
    return resist, react, z

def preprocess_data(v_list, i_list, t, average=True):
    if average:
        avg_v = np.vstack(v_list).mean(axis=0)
        avg_i = np.vstack(i_list).mean(axis=0)
    else:
        avg_v = np.array(v_list)
        avg_i = np.array(i_list)
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
    return ds_v, ds_i, ds_t

def moving_average(x,window_size):
    #moving average across data x within a window_size
    half_window_size=int(window_size/2)
    time_len=len(x)
    moving_average_trace=[]
    max_lim=(time_len-window_size/2)
    min_lim=half_window_size
    for i in range(time_len):
        if i >=min_lim and i <=max_lim :
            moving_average_trace.append(np.mean(x[i-half_window_size:i+half_window_size]))
        elif i<min_lim:
            moving_average_trace.append(np.mean(x[0:i+half_window_size]))
        elif i>max_lim:
            moving_average_trace.append(np.mean(x[i-half_window_size:time_len]))

    return np.asarray(moving_average_trace)

moving_avg2 =moving_average


def plot_impedance_trace(imp,freq,moving_avg_wind,fig_idx,sharpness_thr,filtered_method):
    #From VALIENTAE et Al.
    #generate impedance trace over frequency with peak and cutoff frequency detection
    imp=imp/1e6
    plt.plot(freq,imp, label='raw trace')
    
    prominence_factor=1.01
    if filtered_method==1:
       filtered_imp=moving_average(imp,moving_avg_wind)
    elif filtered_method==2:
       start_idx=np.argmin(freq-0.5)
       freq=freq[start_idx:]
       imp=imp[start_idx:]
       filtered_imp=moving_average(imp,moving_avg_wind)

#    filtered_imp = savgol_filter(imp, moving_avg_wind, 1)
    plt.plot(freq,filtered_imp, label='running mean')
    
    plt.ylim([np.min(imp)*0.9,np.max(imp)*1.1])
    idx_max_mag=np.argmax(filtered_imp)
    cen_freq=freq[idx_max_mag]
    #plt.scatter(freq[idx_max_mag], filtered_imp[idx_max_mag], label='Computed center freq', zorder=999, color='r')

    
    left_imp_mean=np.median(filtered_imp[0:idx_max_mag-1])
    right_imp_mean=np.median(filtered_imp[idx_max_mag+1:])
    max_imp=filtered_imp[idx_max_mag]
         
        
    if (left_imp_mean*prominence_factor)>max_imp  or  (right_imp_mean*prominence_factor)>max_imp or cen_freq<0.5 :
        cen_freq=0  
        
    if cen_freq>0:
        res_sharpness=max_imp/filtered_imp[np.argmin(freq-0.5)]
    else:
        res_sharpness=0
        
    if sharpness_thr>res_sharpness:
        cen_freq=0  
    #find cutoff freq(3dB below max)
    if cen_freq>0:
        i_3db_cutoff=np.argmin(abs(filtered_imp-max_imp/np.sqrt(2)))
        freq_3db=freq[i_3db_cutoff]
    else:
        freq_3db=0
    #plt.scatter(freq[i_3db_cutoff], filtered_imp[i_3db_cutoff], label='Computed 3db freq', zorder=9999, color='g')
    _x = np.hstack((filtered_imp[:5], freq[filtered_imp.shape[0]//2:]))
    _y = np.hstack((filtered_imp[:5], filtered_imp[filtered_imp.shape[0]//2:]))
    
    diff = moving_average(np.diff(filtered_imp),moving_avg_wind) / np.diff(freq)
    ddiff = moving_average( np.diff(diff) / np.diff(freq)[:-1],moving_avg_wind)
    
    #plt.twinx()
    #plt.plot(freq[:-1], diff, c='k')
    #plt.plot(freq[:-2], ddiff, c='r')
    #plt.xscale('log')
    plt.xlabel('Frequency[Hz]')
    plt.ylabel('Impedance[MOhms]')
    if cen_freq is not None:
        if freq_3db is not None:
            plt.title('Trial {fig_idx}, '.format(fig_idx=fig_idx)+'Fr={:.2f} Hz, Cutoff Freq={:.2f}Hz, Sharpness={:.2f}'.format(cen_freq,freq_3db,res_sharpness))
        else:
            plt.title('Trial {fig_idx}, '.format(fig_idx=fig_idx)+'Fr={:.2f} Hz, Cutoff Freq=None'.format(cen_freq))
    else:
        plt.title('Trial {fig_idx}, No Resonance')
    #plt.legend(['deriv1','deriv2'])
    
    res_peak = np.clip(freq[:-2][np.argmin(ddiff)], 0.1,np.inf)#hz
    prominence_fact = np.clip(filtered_imp[:-2][np.argmin(ddiff)], 0.1,np.inf)#hz
    dict_peak = {'cen_freq': cen_freq, 'freq_3db': freq_3db, 'res_sharpness': res_sharpness, 'res_peak': res_peak, 'res_peak_height': prominence_fact}
    return dict_peak

def subsample_average(x, width):

    """Downsamples x by averaging `width` points"""
    avg = np.nanmean(x.reshape(-1, width), axis=1)
    return avg

def cal_imp(abf,abf_stimuli, min_freq,  sweep_end_freq):
    ref_freq=np.linspace( min_freq,sweep_end_freq,1000)
    recorded_var=abf.data
    current_data = np.ravel(abf_stimuli.data).reshape(1,-1)
    dataRate=abf.dataRate
    total_time=np.arange(0,recorded_var.shape[1])/dataRate
    n_sample=10000
    adcUnits=abf.adcUnits
    #find the corresponding index for voltage and current in data
    current_idx=1
    voltage_idx=0
    for unit_idx,unit in enumerate(adcUnits):
        if unit=='pA':
            current_idx=unit_idx
        elif unit=='mV':
            voltage_idx=unit_idx
#    hamming_window=np.hamming(len(time))
#    hamming_window=gaussian(len(time), std=len(time)/2)
    #count the number of data sets in one folder and segment data accordingly
    sweepList=abf.sweepList 
    sweepTimesSec=np.asarray(sweepList)*abf.sweepLengthSec
    voltage_array=[]
    current_array=[]
    time_array=[]
    impedance_array=[]
    for sweep_idx in sweepList: 

        start_idx=int(sweepTimesSec[sweep_idx]*dataRate)
        if sweep_idx==sweepList[-1]:
            end_idx=len(total_time)-1
        else:
            end_idx=int(sweepTimesSec[sweep_idx+1]*dataRate)-1
            
        
        voltage=recorded_var[voltage_idx,start_idx:end_idx]
        N=voltage.shape[0]
        width = int(N / n_sample)
        pad = int(width*np.ceil(N/width) - N)
        
        
        voltage_detrend=subsample_average(np.pad(voltage, (pad,0), 'constant', constant_values=np.nan), width)
        current=current_data[voltage_idx,start_idx:end_idx]
        current_detrend=subsample_average(np.pad(current, (pad,0), 'constant', constant_values=np.nan), width)
        voltage_array.append(voltage_detrend)
        current_array.append(current_detrend)
        time=total_time[start_idx:end_idx]-total_time[start_idx]
        time=time[::width]
        time_array.append(time)
        #FFT on voltage and current
        sp_V= np.fft.fft(voltage_detrend)
        
        
        sp_I= np.fft.fft(current_detrend)
        freq = np.fft.fftfreq(len(time), d=time[1])
        half_freq=int(len(time)/2)
        
        #discard frequency above 20Hz and negative frequency
        freq=freq[1:half_freq]
        sp_V=sp_V[1:half_freq]
        sp_I=sp_I[1:half_freq]
        selected_freq=freq<21
        impedance=np.abs(sp_V[selected_freq]/sp_I[selected_freq])
        
        f_v=interp1d(freq[selected_freq],sp_V[selected_freq])
        f_i=interp1d(freq[selected_freq],sp_I[selected_freq])
        f_imp=interp1d(freq[selected_freq],impedance,fill_value="extrapolate")
        interpolated_trace=f_imp(ref_freq)
        impedance_array.append(interpolated_trace)
#        plt.figure()
#        plt.plot(time,recorded_var[voltage_idx,start_idx:end_idx])
        

    return impedance_array,ref_freq,voltage_array,current_array,time_array

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
        ds_v, ds_i, ds_t = v, i, t
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
        return resistance[low_ind:high_ind], reactance[low_ind:high_ind], xf[low_ind:high_ind]

def generate_abf_array(file_path, stimuli_abf, moving_avg_win_in, max_freq, min_freq):
    file_path = os.path.join(root,filename)
    abf = pyabf.ABF(file_path)
    
    print(abf.abfID + ' loaded')
    abf_name = np.vstack([abf.abfID,abf.abfID, abf.abfID, abf.abfID, abf.abfID])
    abf_label = np.vstack(['resist','react', 'freq', 'resist running avg', 'react running avg'])
    abf_feat = analyze_abf_chirp(abf, stimuli_abf, average, min_freq, max_freq)
    #VALIENTE ANALYSIS
    plt.clf()
    #test = cal_imp(abf, stimuli_abf, min_freq, max_freq)
    peaks_dict = plot_impedance_trace(abf_feat[0], abf_feat[2],moving_avg_win_in*2, 1, 0, 1)
    plt.pause(0.1)
    running_mean_resist =  moving_avg2(abf_feat[0], moving_avg_win_in)
    running_mean_react =  moving_avg2(abf_feat[1], moving_avg_win_in)
    tpeaks = find_peak(abf_feat[2], running_mean_resist)
    tpeaks[0].update(peaks_dict)
    temp = pd.DataFrame().from_dict(tpeaks[0])
    temp['id'] = np.full(temp.index.values.shape[0], abf.abfID)
    temp['width'] = np.full(temp.index.values.shape[0], tpeaks[1])
    
    abf_feat = np.vstack((abf_feat, running_mean_resist))
    abf_feat = np.vstack((abf_feat, running_mean_react))
    abf_ar = np.hstack((abf_name, abf_label, abf_feat))
    abf_ar = np.hstack((abf_ar, np.vstack([np.full(len_f  - abf_ar.shape[1], np.nan), np.full(len_f  -abf_ar.shape[1], np.nan), np.full(len_f  -abf_ar.shape[1], np.nan), np.full(len_f  -abf_ar.shape[1], np.nan), np.full(len_f  -abf_ar.shape[1], np.nan)])))
    return abf_ar, temp


## Ask the opening Q's
down_rate = 20000
root = tk.Tk()
root.withdraw()
files = filedialog.askdirectory(
                                   title='Select dir File'
                                   )

root_fold = files
print("Stimuli options")
stim_names = [os.path.basename(x) for x in atf_files]
stim_file = atf_files
for x,name in enumerate(stim_names):
    print(str(x+1)+ ". " + name)

stim_num = input("Please Enter the number of the stimuli used:")
try: 
    stim_num  = int(stim_num) - 1
    stimuli = stim_file[stim_num]
except:
    stimuli = stim_file[0]



tag = input("tag to apply output to files: ")
try: 
    tag = str(tag)
except:
    tag = ""

average_str = input("Enter Average sweep input? (y/n) (Otherwise Curve output is averaged):")
try: 
    if average_str == "y" or average_str=="Y":
        average="input"
    else:
        average="output"
except:
    average="input"

lowerlim = input("Enter the Lower Cutoff for Freq to include in output [in Hz] (recommended 0.5Hz): ")
upperlim = input("Enter the Upper Cutoff for Freq to include in output [in Hz] (recommended 20Hz): ")

try: 
    min_freq = float(lowerlim)
    
except:
    min_freq=0.5
    

try:
    max_freq = float(upperlim)-0.5
except:
    max_freq=20


moving_avg_win_in = input("Enter a window for calc of the moving averages (in n+1 points) (default 50+1):")
try:
    moving_avg_win_in = int(moving_avg_win_in)+1
except:
    moving_avg_win_in = 51
# lowerlim = input("Enter the time to begin analysis [in s] (recommended 0.78): ")
# upperlim = input("Enter the time to finish analysis [in s] (recommended 49.21): ")

# try: 
#     start = float(lowerlim)
#     end = float(upperlim)
# except:
#     pass


print("loading stimuli")
extension = stimuli.split(".")[-1]
if 'atf' in extension:
    stimuli_abf = pyabf.ATF(stimuli)
elif 'abf' in extension:
    stimuli_abf = pyabf.ABF(stimuli)
print("stimuli loaded")
len_f = 1000
peaks = []
full = np.full(len_f , np.nan)
for root,dir,fileList in os.walk(files):
 for filename in fileList:
    if filename.endswith(".abf"):
        try:
            abf_ar, temp_peak = generate_abf_array(filename, stimuli_abf, moving_avg_win_in, max_freq, min_freq)
            full = np.vstack((full, abf_ar))
            peaks.append(temp_peak)
        except Exception as e:
           print("issue processing {filename}")
           print(e)

peaks = pd.concat(peaks, axis=0)
np.savetxt(root+'/CHIRP.csv', full, delimiter=",", fmt='%s')
peaks.drop_duplicates('id').to_csv(root+"/CHIRP_resist_peaks_no_dupe.csv")
