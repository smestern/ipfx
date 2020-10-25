
from watchdog.events import RegexMatchingEventHandler
from watchdog.observers import Observer
import os
import glob
import pandas as pd
from scipy import stats
import pyabf
import time
import numpy as np
class LiveQC(RegexMatchingEventHandler):

    def __init__(self, settingsFile):
        super().__init__(regexes=[".*\.abf"], ignore_directories=True, case_sensitive=False)

        self.settingsFile = settingsFile

    def on_created(self, event):

        print(f"Running QC for file {event.src_path}.")

        base, _ = os.path.splitext(event.src_path)
        
        i = 0
        while i < 101:
            i+=1
            time.sleep(5)
            try:
                dataX, dataY, dataC = loadABF(event.src_path)
                break
            except:
                continue
        
        try:
            qc = run_qc(dataY, dataC)
            rms_okay, drif_okay = eval_qc(qc)
            print("==== QC RESULTS ====")
            if rms_okay:
                print(f"[QC] {event.src_path} passes RMS QC")
            else:
                print(f"[QC] {event.src_path} fails RMS QC [PLEASE DOUBLE CHECK]")

            if drif_okay:
                print(f"[QC] {event.src_path} passes Voltage QC")
            else:
                print(f"[QC] {event.src_path} fails Voltage QC [PLEASE DOUBLE CHECK]")
        
        except Exception as e:
            print(f"Ignoring exception {e}.")

def eval_qc(a):
    rms_okay = True
    drif_okay = True
    if a[0] > 7:
       rms_okay = False

    if a[2] > 5:
        drift_okay = False
    return rms_okay, drif_okay

def loadABF(file_path, return_obj=False):
    '''
    Employs pyABF to generate numpy arrays of the ABF data. Optionally returns abf object.
    Same I/O as loadNWB
    '''
    abf = pyabf.ABF(file_path)
    dataX = []
    dataY = []
    dataC = []
    for sweep in abf.sweepList:
        abf.setSweep(sweep)
        tempX = abf.sweepX
        tempY = abf.sweepY
        tempC = abf.sweepC
        dataX.append(tempX)
        dataY.append(tempY)
        dataC.append(tempC)
    npdataX = np.vstack(dataX)
    npdataY = np.vstack(dataY)
    npdataC = np.vstack(dataC)

    if return_obj == True:

        return npdataX, npdataY, npdataC, abf
    else:

        return npdataX, npdataY, npdataC

    ##Final return incase if statement fails somehow
    return npdataX, npdataY, npdataC

def find_zero(realC):
    #expects 1d array
    zero_ind = np.where(realC == 0)[0]
    return zero_ind

def compute_vm_drift(realY, zero_ind):
    sweep_wise_mean = np.mean(realY[:,zero_ind], axis=1)
    mean_drift = np.abs(np.amax(sweep_wise_mean) - np.amin(sweep_wise_mean))
    abs_drift = np.abs(np.amax(realY[:,zero_ind]) - np.amin(realY[:,zero_ind]))
   
    return mean_drift, abs_drift


def compute_rms(realY, zero_ind):
    mean = np.mean(realY[:,zero_ind], axis=1)
    rms = []
    for x in np.arange(mean.shape[0]):
        temp = np.sqrt(np.mean(np.square(realY[x,zero_ind] - mean[x])))
        rms = np.hstack((rms, temp))
    full_mean = np.mean(rms)
    return full_mean, np.amax(rms)

def run_qc(realY, realC):
    zero_ind = find_zero(realC[0,:])
    mean_rms, max_rms = compute_rms(realY, zero_ind)
    mean_drift, max_drift = compute_vm_drift(realY, zero_ind)
    return [mean_rms, max_rms, mean_drift, max_drift]


def main():
    _dir = os.path.dirname(__file__)
    _path = glob.glob(_dir +'//..//data_and_results//HYP_CELL_NWB//Naive//*.nwb')
    full_qc = [0,0,0,0]
    for fp in _path:
        realX, realY, realC = loadNWB(fp)
        temp_qc = run_qc(realY, realC)
        full_qc = np.vstack((full_qc, temp_qc))
    df = pd.DataFrame(data=full_qc[1:,:], columns=['Mean RMS', 'Max RMS', 'Mean Drift', 'Max Drift'], index=_path)
    df.to_csv('qc.csv')
    stats = []
    for col in df.columns.values:
        stats.append(df[col].quantile(0.1))
    qc_stats = pd.DataFrame(data=stats, index=['10 percentile Mean RMS', '10 percentile Max RMS', ' 10 percentile Mean Drift', ' 10 percentile Max Drift'])
    qc_stats.to_csv('qc_stats.csv')

if __name__ == "__main__": 
    main()
