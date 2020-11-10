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

def remove_sweeps(nwb_file, qcsweeps):
  qcsweeps = np.asarray(qcsweeps) #If given a list convert to array
  file_path = nwb_file
  print(f"QC'ing {file_path}")
  with h5py.File(file_path,  "r") as f:
            item = f['acquisition']
            sweeps = item.keys()
            print(sweeps)
  qc_names = []
  for x in qcsweeps:
          if x < 10:
              qc_names.append(f"index_0{x}") ##This is for if the file has under 100 sweeps, otherwise the names will be something like, index_00X
          else:
              qc_names.append(f"index_{x}")
  print(qc_names)
  with h5py.File(file_path,  "a") as f:
        item = f['acquisition'] ##Delete the response recording
        for p in qc_names:
              try:
                  del item[p] #For whatever reason these deletes try to do it twice ignore the second error message
                  print(f'deleted {p}')
              except:
                  print(f'{p} delete fail')
        item = f['stimulus'] #next delete the stimset
        for p in qc_names:
              try:
                  del item[p]
                  print(f'deleted {p}')
              except:
                  print(f'{p} delete fail')
        print(item.keys())
        item = f['general']['intracellular_ephys']['sweep_table'] #next delete the references in the sweep table, or else the nwbs may break analysis
        ## Since IPFX may go looking for sweeps that are absent
        for key, value in item.items():
              array = value[()]
              ind = np.arange(0, len(array))
              
              bool_mask = np.in1d(ind,qcsweeps, invert=True)
              new_data = array[bool_mask]
              try:
                del item[key]
                item[key] = new_data
                print(f'deleted and rewrote {key}')
              except: 
                print(f'{key} delete fail')

def main():
    NHPPath = "C:\\Users\\SMest\\Downloads\\Final_LP_selection"
    sweep_qc = pd.read_csv("C:\\Users\\SMest\\Downloads\\QCed_NHP_cells_website.csv", index_col=0)
    #NHPPath = "C://Users//SMest//Documents//NHP"
    new_path =  "C:\\Users\\SMest\\Documents\\New folder3\\"
    #sweep_qc = pd.read_csv("C:\\Users\\SMest\\Documents\\clustering-data\\sweep_qc.csv", index_col=0)
    protocol = []
    cell_list = sweep_qc.index.values
    
    #NHPPath = "C:\\Users\\SMest\\Documents\\New folder2"
    for r, celldir, f in os.walk(NHPPath):
              
              for c in celldir: ##Walks through each folder (cell folder) in the root folder

                   c = os.path.join(r, c) ##loads the subdirectory path
                   shutil.copy("C:\\Users\\SMest\\Documents\\NHP\\default.json",c)
              for file in f:
                  if '.nwb' in file:
                   try:
                       cell_name = file.split('.')[0]
                       cell_qc = np.where(sweep_qc.loc[cell_name].values==0)[0]
                       file_path = os.path.join(r,file)
                       print(f"Converting {cell_name}")
                       remove_sweeps(file_path, cell_qc)
                   except:
                     print('fail')


if __name__ == "__main__":
    main()
