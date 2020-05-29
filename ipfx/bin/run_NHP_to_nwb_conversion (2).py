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

    NHPPath = "C://Users//SMest//Documents//NHP"
    sweep_qc = pd.read_csv("C:\\Users\\SMest\\Documents\\clustering-data\\sweep_qc.csv", index_col=0)
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
                                abf.sweepList
                                for sweepNumber in cell_qc:
                                    abf.setSweep(sweepNumber)
                                    
                                
                                    try:
                                        current = abf.sweepC[np.nonzero(abf.sweepC)[0][0]]
                                    except:
                                        current = 0
                                    out = np.hstack((out, current))
                                out = np.hstack((cell_name, out))
                                stimlist.append(out)

                                #cell_qc = np.nonzero(sweep_qc.loc[cell_name].values)[0] + 1
                                #cell_qc_s = np.array2string(cell_qc, threshold=99, separator=',')
                                #cell_qc_s = cell_qc_s.split('[')[-1].split(']')[0]
                                #pyperclip.copy(cell_qc_s)
                                #print(cell_qc_s)
                                #abf.launchInClampFit()
                                #time.sleep(3)
                                #with key.pressed(pynput.keyboard.Key.cmd):
                                #       key.press(pynput.keyboard.Key.up)
                                #       key.release(pynput.keyboard.Key.up)
                                #time.sleep(3)
                                #key.release(pynput.keyboard.Key.cmd)
                                #mouse.position = (609, 347)
                                #mouse.press(pynput.mouse.Button.left)
                                #mouse.release(pynput.mouse.Button.left)
                                #time.sleep(1)
                                #mouse.position = (76, 31)
                                #mouse.press(pynput.mouse.Button.left)
                                #mouse.release(pynput.mouse.Button.left)
                                #time.sleep(1)
                                #mouse.position = (141, 505)
                                
                                #time.sleep(1)
                                
                                #mouse.press(pynput.mouse.Button.left)
                                #mouse.release(pynput.mouse.Button.left)
                                #time.sleep(1)
                                #mouse.position = (831, 610)
                                #mouse.press(pynput.mouse.Button.left)
                                #mouse.release(pynput.mouse.Button.left)
                                #time.sleep(1)
                                #mouse.position = (928, 634)
                                #time.sleep(1)
                                #mouse.click(pynput.mouse.Button.left, 1)
                                #time.sleep(1)
                                #for p in np.arange(4):
                                #    time.sleep(1)
                                #    key.press(pynput.keyboard.Key.backspace)
                                #    key.release(pynput.keyboard.Key.backspace)
                                
                                
                                #time.sleep(1)
                                #key.type(cell_qc_s)
                                #for p in np.arange(100):
                                #    time.sleep(0.05)
                                #    key.press(pynput.keyboard.Key.left)
                                #    key.release(pynput.keyboard.Key.left)
                                #key.press(pynput.keyboard.Key.right)
                                #key.release(pynput.keyboard.Key.right)
                                #key.press(pynput.keyboard.Key.backspace)
                                #key.release(pynput.keyboard.Key.backspace)
                                #time.sleep(1)
                                #mouse.position = (863, 687)
                                #mouse.click(pynput.mouse.Button.left, 1)
                                #time.sleep(1)
                                #mouse.position = (11, 32)
                                #mouse.click(pynput.mouse.Button.left, 1)
                                #time.sleep(1)
                                #mouse.position = (85, 277)
                                #mouse.click(pynput.mouse.Button.left, 1)
                                #time.sleep(1)
                                #key.type(cell_name)
                                #key.press(pynput.keyboard.Key.enter)
                                #key.release(pynput.keyboard.Key.enter)
                                #key.press(pynput.keyboard.Key.enter)
                                #key.release(pynput.keyboard.Key.enter)
                                #key.press(pynput.keyboard.Key.enter)
                                #key.release(pynput.keyboard.Key.enter)
                                #key.press(pynput.keyboard.Key.enter)
                                #key.release(pynput.keyboard.Key.enter)
                                #time.sleep(1)
                                #mouse.position = (1891, 11)
                                #mouse.click(pynput.mouse.Button.left, 1)
                                
                                #time.sleep(10)
                  except: 
                       print("fail")
    NHPPath = "C:\\Users\\SMest\\Documents\\New folder"
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
