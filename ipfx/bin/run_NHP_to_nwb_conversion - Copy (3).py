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
    #for r, celldir, f in os.walk(NHPPath):
              
    #          for c in celldir: ##Walks through each folder (cell folder) in the root folder

    #               c = os.path.join(r, c) ##loads the subdirectory path
    #               shutil.copy("C:\\Users\\SMest\\Documents\\NHP\\default.json",c)
    #          for file in f:
    #              try:
    #                  if '.abf' in file and '.png' not in file:
    #                   cell_name = r.split('\\')[-1]
    #                   if cell_name in cell_list:
    #                        file_path = os.path.join(r,file)
    #                        cell_qc = np.nonzero(sweep_qc.loc[cell_name].values)[0]
            
    #                        abf = pyabf.ABF(file_path)
    #                        if '1000' in abf.protocol:
    #                            for x in np.arange(abf.sweepCount):
    #                                abf.setSweep(x)
    #                                if x == 0:
    #                                    sweep_data = abf.sweepY
    #                                else:
    #                                    sweep_data = np.vstack((sweep_data, abf.sweepY))
    #                            mask_sweep_data = sweep_data[cell_qc,:]
    #                            pyabf.abfWriter.writeABF1(mask_sweep_data, 'C:\\Users\\SMest\\Documents\\NHP-QC\\' + cell_name +'.abf', abf.dataRate, abf.sweepUnitsY)
    #              except:
    #                   print("fail")
    NHPPath = "C:\\Users\\SMest\\Documents\\NHP_MARM\\210204_Marm"
    for r, celldir, f in os.walk(NHPPath):
              
              #for c in celldir: ##Walks through each folder (cell folder) in the root folder

                   #c = os.path.join(r, c) ##loads the subdirectory path
                   #shutil.copy("C:\\Users\\SMest\\Documents\\NHP\\default.json",c)
              for file in f:
                  if '.abf' in file:
                   file_path = os.path.join(r,file)
                   print(f"Converting {c}")
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
