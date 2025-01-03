#!/bin/env python
import shutil
import os
import argparse
import logging
log = logging.getLogger(__name__)
import pyabf
from ipfx.x_to_nwb.ABFConverter import ABFConverter
from ipfx.x_to_nwb.DatConverter import DatConverter
import numpy as np
import pandas as pd
import pynput
import time
import pyperclip
def convert(inFileOrFolder, overwrite=False, fileType=None, outputMetadata=False, outputFeedbackChannel=False, multipleGroupsPerFile=False, compression=True):
    """
    Convert the given file to a NeuroDataWithoutBorders file using pynwb

    Supported fileformats:
        - ABF v2 files created by Clampex
        - DAT files created by Patchmaster v2x90

    :param inFileOrFolder: path to a file or folder
    :param overwrite: overwrite output file, defaults to `False`
    :param fileType: file type to be converted, must be passed iff `inFileOrFolder` refers to a folder
    :param outputMetadata: output metadata of the file, helpful for debugging
    :param outputFeedbackChannel: Output ADC data which stems from stimulus feedback channels (ignored for DAT files)
    :param multipleGroupsPerFile: Write all Groups in the DAT file into one NWB
                                  file. By default we create one NWB per Group (ignored for ABF files).
    :param compression: Toggle compression for HDF5 datasets

    :return: path of the created NWB file
    """

    if not os.path.exists(inFileOrFolder):
        raise ValueError(f"The file {inFileOrFolder} does not exist.")

    if os.path.isfile(inFileOrFolder):
        root, ext = os.path.splitext(inFileOrFolder)
    if os.path.isdir(inFileOrFolder):
        if not fileType:
            raise ValueError("Missing fileType when passing a folder")

        inFileOrFolder = os.path.normpath(inFileOrFolder)
        inFileOrFolder = os.path.realpath(inFileOrFolder)

        ext = fileType
        root = os.path.join(inFileOrFolder, "..",
                            os.path.basename(inFileOrFolder))

    outFile = root + ".nwb"

    if not outputMetadata and os.path.exists(outFile):
        if overwrite:
            os.remove(outFile)
        else:
            raise ValueError(f"The output file {outFile} does already exist.")

    if ext == ".abf":
        if outputMetadata:
            ABFConverter.outputMetadata(inFileOrFolder)
        else:
            ABFConverter(inFileOrFolder, outFile, outputFeedbackChannel=outputFeedbackChannel, compression=compression)
    elif ext == ".dat":
        if outputMetadata:
            DatConverter.outputMetadata(inFileOrFolder)
        else:
            DatConverter(inFileOrFolder, outFile, multipleGroupsPerFile=multipleGroupsPerFile, compression=compression)

    else:
        raise ValueError(f"The extension {ext} is currently not supported.")

    return outFile


def main():

    NHPPath = "C:\\Users\\SMest\\Downloads\\Final_LP_selection"
    new_path =  "C:\\Users\\SMest\\Documents\\New folder3\\"
    sweep_qc = pd.read_csv("C:\\Users\\SMest\\Downloads\\QCed_NHP_cells_website.csv", index_col=0)
    protocol = []
    cell_list = sweep_qc.index.values
    mouse = pynput.mouse.Controller()
    key = pynput.keyboard.Controller()
    stimlist = []
    for r, celldir, f in os.walk(NHPPath):
              
              for c in celldir: ##Walks through each folder (cell folder) in the root folder

                   c = os.path.join(r, c) ##loads the subdirectory path
                   shutil.copy("C:\\Users\\SMest\\Documents\\NHP\\default.json",c)
              for file in f:
                  try:
                      if '.abf' in file and '.png' not in file:
                       cell_name = r.split('\\')[-1]
                       if cell_name in cell_list:
                            file_path = os.path.join(r,file)
                            
                            abf = pyabf.ABF(file_path)

                            if '1000' in abf.protocol:
                                out = 0
                                cell_qc = sweep_qc.loc[cell_name].values.astype(np.int32).ravel()
                                
                                for sweepNumber in abf.sweepList:
                                    abf.setSweep(sweepNumber)
                                    
                                
                                    try:
                                        current = abf.sweepC[np.nonzero(abf.sweepC)[0][0]]
                                    except:
                                        current = 0
                                    out = np.hstack((out, current))
                                out = np.hstack((cell_name, out))
                                stimlist.append(out)
                                #shutil.copy(file_path, (new_path+cell_name+'.abf'))
                  except: 
                       print("fail")
    NHPPath = "C:\\Users\\SMest\\Documents\\New folder2"
    maxlen = max(stimlist,key=len).shape[0]
    for x,l in enumerate(stimlist):
        if x==0:
            astim =  np.hstack((l,np.full(maxlen - l.shape[0], np.nan)))
        else:
            astim = np.vstack((astim, np.hstack((l,np.full(maxlen - l.shape[0], np.nan)))))
    np.savetxt('stimlist.csv', astim , delimiter=",", fmt='%s')
    for r, celldir, f in os.walk(NHPPath):
              
              for c in celldir: ##Walks through each folder (cell folder) in the root folder

                   c = os.path.join(r, c) ##loads the subdirectory path
                   shutil.copy("C:\\Users\\SMest\\Documents\\NHP\\default.json",c)
              for file in f:
                  if '.abf' in file:
                   file_path = os.path.join(r,file)
                   #print(f"Converting {c}")
                   convert(file_path
                           ,
                        overwrite=True,
                        fileType='.abf',
                        outputMetadata=False,
                        outputFeedbackChannel=False,
                        multipleGroupsPerFile=True,
                        compression=True)
    np.savetxt("C:\\Users\\SMest\\Documents\\NHP\\protocol.csv", protocol, fmt="%s") 

if __name__ == "__main__":
    main()
