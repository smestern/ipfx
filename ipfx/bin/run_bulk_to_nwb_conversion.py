#!/bin/env python
import shutil
import os
import h5py
import argparse
import logging
import pynwb
import json
log = logging.getLogger(__name__)
import pyabf
from ipfx.x_to_nwb.ABFConverter import ABFConverter
from ipfx.x_to_nwb.DatConverter import DatConverter
import numpy as np
import pandas as pd
import collections
from hdmf.utils import docval, popargs
from pynwb import NWBFile, register_class, load_namespaces, NWBHDF5IO, CORE_NAMESPACE, get_class
from pynwb.spec import NWBNamespaceBuilder, NWBGroupSpec, NWBAttributeSpec
from pynwb.file import LabMetaData


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



def confirm_metadata(file, mjson, meta_field=True):
    """
    Function Takes an input NWB, and INPUT json file(s). Checks to see if keys within the NWB that match JSON keys have matching content, if not, overwrites. Sometimes metadata
    added to the json file is ignored by the ABFCONVERTER. Adds novel metadata keys as NWB extensions safely using pyNWB. Overwrites using h5py.
    Takes:
    file: path to NWB (hdf5) file,
    mjson: path to json file to be injected into file.
    meta_field: If true, all novel data is filed under a new group in the NWB file titled 'metadata' otherwise metadata is placed in base group
    returns:
    file: path to NWB file.
    """
    def loadJSON(filename):
        if isinstance(filename, (list, np.ndarray)):
            full_dict = {}
            for js in filename:
                with open(js) as fh:
                     full_dict.update(json.load(fh))
            return full_dict
        else:
           with open(filename) as fh:
                return json.load(fh)
    def _h5_merge(dict1, dict2):
        ''' Recursively merges the input'''

        result = dict1

        for key, value in dict2.items():
            if isinstance(value, collections.Mapping):
                _h5_merge(result.get(key, {}), value)
            else:
                result[key][...] = dict2[key]

        return result
 

    metadata = loadJSON(mjson)
    with h5py.File(file,  "r+") as f: ##Has to be opened with h5py as pyNWB does not support overwrite
        NWB_f = f
        nwb_keys = list(NWB_f.keys())
        meta_keys = list(metadata.keys())
        overlap_keys = np.intersect1d(nwb_keys,meta_keys) ##Look for overlapping and overwrite
        for key in overlap_keys:
            if isinstance(metadata[key], dict):
                d = _h5_merge(f[key], metadata[key]) ## if its a dict instance, we begin the process of merging
                ##recursively 
            else:
                f[key][...] = metadata[key] ## Otherwise just overwrite. 
        novel_keys = np.setdiff1d(meta_keys, overlap_keys) ##Grab the novel keys for later
    n_metadata = {key: metadata[key] for key in novel_keys}
    ##Now close the nwb and open with pynwb
    with pynwb.NWBHDF5IO(file,  mode="r+") as f_io:
        ### Now add novel data using pynwb in a way thats a lot less brute force, and way more nwb friendly
        f = f_io.read()
        NWB_f = f
        if False:
            ##Currently not working ##Class is compiled but attributes are not written to files
            meta_class = build_settings(n_metadata)
            test = get_class('MetaData', 'NHP')

            nwb_meta = test(name='meta', experiment_id=int(12), test='sfesfesf')
            NWB_f.add_lab_meta_data(nwb_meta)
        else:
            ## For now just dump into scratch ##Goes against NWB Conventions however
            for key, value in n_metadata.items():
                if isinstance(value, dict):
                    cont = dict_to_list(value)
                    for x in cont:
                       NWB_f.add_scratch([x[1]], name=str(x[0]), notes=str(key))
                else:
                    NWB_f.add_scratch([value], name=str(key), notes="null")
                    
       
        
        f_io.write(NWB_f)
        f_io.close()



def dict_to_nwbdata(dict1):
        list = []
        for key, value in dict1.items():
            if isinstance(value, (dict)):
                list.append(dict_to_nwbdata(value))
            else:
                list.append(pynwb.core.NWBData(name=str(key), data=[value]))
        return list
def dict_to_list(dict1):
        list = []
        for key, value in dict1.items():
            if isinstance(value, (dict)):
                list.append(dict_to_list(value))
            else:
                list.append((str(key), str(value)))
        return list

def build_settings(dict):
    """ WIP builds metadata based on passed in dict Does not support dict(s) deeper than 1 ATM"""
    
    # Settings:
    neurodata_type = 'MetaData'
    prefix = 'NHP'
    outdir = './'
    extension_doc = 'lab metadata extension'


    metadata_ext_group_spec = NWBGroupSpec(
        neurodata_type_def=neurodata_type,
        neurodata_type_inc='LabMetaData',
        doc=extension_doc,
        attributes=[NWBAttributeSpec(name='experiment_id', dtype='int', doc='HW'), NWBAttributeSpec(name='test', dtype='text', doc='HW')])


    #Export spec:
    ext_source = '%s_extension.yaml' % prefix
    ns_path = '%s_namespace.yaml' % prefix
    ns_builder = NWBNamespaceBuilder(extension_doc, prefix, version=str(1))
    ns_builder.add_spec(ext_source, metadata_ext_group_spec)
    ns_builder.export(ns_path, outdir=outdir)

     #Read spec and load namespace:
    ns_abs_path = os.path.join(outdir, ns_path)
    load_namespaces(ns_abs_path)


    
    class MetaData(LabMetaData):
        __nwbfields__ = ('experiment_id','test')

        @docval({'name': 'name', 'type': str, 'doc': 'name'},
                {'name': 'experiment_id', 'type': int, 'doc': 'HW'},
                {'name': 'test', 'type': str, 'doc': 'HW'})
        def __init__(self, **kwargs):
            name, ophys_experiment_id, test = popargs('name', 'experiment_id', 'test', kwargs)
            super(OphysBehaviorMetaData, self).__init__(name=name)
            self.experiment_id = experiment_id
            self.test = test


    register_class('MetaData', prefix,MetaData)
    return MetaData

            


    

def main():
    parser = argparse.ArgumentParser()

    common_group = parser.add_argument_group(title="Common", description="Options which are applicable to both ABF and DAT files")
    abf_group = parser.add_argument_group(title="ABF", description="Options which are applicable to ABF")
    dat_group = parser.add_argument_group(title="DAT", description="Options which are applicable to DAT")

    feature_parser = common_group.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--compression', dest='compression', action='store_true', help="Enable compression for HDF5 datasets (default).")
    feature_parser.add_argument('--no-compression', dest='compression', action='store_false', help="Disable compression for HDF5 datasets.")
    parser.set_defaults(compression=True)

    common_group.add_argument("--overwrite", action="store_true", default=False,
                               help="Overwrite the output NWB file")
    common_group.add_argument("--outputMetadata", action="store_true", default=False,
                               help="Helper for debugging which outputs HTML/TXT files with the metadata contents of the files.")
    common_group.add_argument("--log", type=str, help="Log level for debugging, defaults to the root logger's value.")
    common_group.add_argument("filesOrFolders", nargs="+",
                               help="List of ABF files/folders to convert.")
    common_group.add_argument("--additionalMetadata", default=None,
                              help="Pointed towards additonal JSON file which will be added to each NWB")
    common_group.add_argument("--amplifierSettings", default=None,
                              help="Pointed towards additonal JSON file which will be added to each NWB")

    abf_group.add_argument("--protocolDir", type=str,
                            help=("Disc location where custom waveforms in ATF format are stored."))
    abf_group.add_argument("--fileType", type=str, default=None, choices=[".abf"],
                            help=("Type of the files to convert (only required if passing folders)."))
    abf_group.add_argument("--outputFeedbackChannel", action="store_true", default=False,
                        help="Output ADC data to the NWB file which stems from stimulus feedback channels.")
    abf_group.add_argument("--realDataChannel", type=str, action="append",
                        help=f"Define additional channels which hold non-feedback channel data. The default is {ABFConverter.adcNamesWithRealData}.")

    dat_group.add_argument("--multipleGroupsPerFile", action="store_true", default=False,
                           help="Write all Groups from a DAT file into a single NWB file. By default we create one NWB file per Group.")

    args = parser.parse_args()

    if args.log:
        numeric_level = getattr(logging, args.log.upper(), None)

        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {args.log}")

        logger = logging.getLogger()
        logger.setLevel(numeric_level)

    if args.protocolDir:
        if not os.path.exists(args.protocolDir):
            raise ValueError("Protocol directory does not exist")

        ABFConverter.protocolStorageDir = args.protocolDir

    if args.realDataChannel:
        ABFConverter.adcNamesWithRealData.append(args.realDataChannel)

    
    root_path = args.filesOrFolders
    if args.additionalMetadata is not None:
        if os.path.isfile(args.additionalMetadata):
            meta = args.additionalMetadata
            bmeta = True
        else:
            bmeta = False
    else:
        bmeta = False

    for path in root_path:
        for r, celldir, f in os.walk(path):
              
              for c in celldir: ##Walks through each folder (cell folder) in the root folder

                  c = os.path.join(r, c) ##loads the subdirectory path
                  ls = os.listdir(c) ##Lists the files in the subdir
                  abf_pres = np.any(['.abf' in x for x in ls]) #Looks for the presence of at least one abf file in the folder (does not check subfolders)
                  if abf_pres:
                       if bmeta == True: ##If the user provided an additonal json file, we copy that into the subfolder
                            shutil.copy(meta,c) 
                            
                       print(f"Converting {c}")
                       nwb = convert(c,
                                overwrite=True,
                                fileType='.abf',
                                outputMetadata=args.outputMetadata,
                                outputFeedbackChannel=args.outputFeedbackChannel,
                                multipleGroupsPerFile=True,
                                compression=args.compression)

                       confirm_metadata(nwb,meta)
                       
                       os.remove(os.path.join(c,os.path.basename(meta)))
     

if __name__ == "__main__":
    main()
