#!/bin/env python
import shutil
import os
import argparse
import logging
log = logging.getLogger(__name__)
import pyabf
from ipfx.x_to_nwb.ABFConverter import ABFConverter
from ipfx.x_to_nwb.DatConverter import DatConverter
import h5py
import numpy as np
import pandas as pd
import pynwb
import pynput
import time
import pyperclip


def main():

    NHPPath = "C://Users//SMest//Documents//NHP"
    new_path =  "C:\\Users\\SMest\\Documents\\New folder3\\"
    sweep_qc = pd.read_csv("C:\\Users\\SMest\\Documents\\clustering-data\\sweep_qc.csv", index_col=0)
    protocol = []
    cell_list = sweep_qc.index.values
    
    NHPPath = "C:\\Users\\SMest\\Documents\\New folder2"
    for r, celldir, f in os.walk(NHPPath):
              
              for c in celldir: ##Walks through each folder (cell folder) in the root folder

                   c = os.path.join(r, c) ##loads the subdirectory path
                   shutil.copy("C:\\Users\\SMest\\Documents\\NHP\\default.json",c)
              for file in f:
                  if '.nwb' in file:
                   cell_name = file.split('.')[0]
                   cell_qc = np.where(sweep_qc.loc[cell_name].values==0)[0]
                   file_path = os.path.join(r,file)
                   print(f"Converting {cell_name}")
                   with h5py.File(file_path,  "r") as f:
                            item = f['acquisition']
                            sweeps = item.keys()
                            print(sweeps)
                   qc_names = []
                   for x in cell_qc:
                      if x < 10:
                       qc_names.append(f"index_0{x}")
                      else:
                          qc_names.append(f"index_{x}")
                   
                   #new_acq = {key: nwb_file.acquisition[key] for key in qc_names}
                   #new_stim = {key: nwb_file.stimulus[key] for key in qc_names}
                   with h5py.File(file_path,  "a") as f:
                       item = f['acquisition']
                       for p in qc_names:
                           try:
                            del item[p]
                           except:
                               print('del fail')
                       item = f['stimulus']
                       for p in qc_names:
                           try:
                            del item[p]
                           except:
                               print('del fail')
                       print(item.keys())
                   #nwb_io['acquisition'] = new_acq
                   #nwb_file.stimulus = new_stim
                 


if __name__ == "__main__":
    main()
