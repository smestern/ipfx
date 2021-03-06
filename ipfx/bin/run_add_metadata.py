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
import pandas as pd
import collections
from hdmf.utils import docval, popargs
from pynwb import NWBFile, register_class, load_namespaces, NWBHDF5IO, CORE_NAMESPACE, get_class
from pynwb.spec import NWBNamespaceBuilder, NWBGroupSpec, NWBAttributeSpec
from pynwb.file import LabMetaData

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
                try:
                    result[key][...] = dict2[key]
                except:
                    result[key]= dict2[key] 
        return result




def confirm_metadata(file, metadata=None, mjson=None, meta_field=True):
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
    
 
    if mjson is not None:
        metadata = loadJSON(mjson)
    with h5py.File(file,  "r+") as f: ##Has to be opened with h5py as pyNWB does not support overwrite
        NWB_f = f
        nwb_keys = list(NWB_f.keys())
        meta_keys = list(metadata.keys())
        overlap_keys = np.intersect1d(nwb_keys,meta_keys) ##Look for overlapping and overwrite
        for key in meta_keys:
            if key in overlap_keys:
                if isinstance(metadata[key], dict):
                    d = _h5_merge(f[key], metadata[key]) ## if its a dict instance, we begin the process of merging
                    ##recursively 
                else:
                    f[key][...] = metadata[key] ## Otherwise just overwrite.
            else:
                f[key] = metadata[key] ## Otherwise just overwrite.
        novel_keys = np.setdiff1d(meta_keys, overlap_keys) ##Grab the novel keys for later
       

    n_metadata = {key: metadata[key] for key in novel_keys}
    ##Now close the nwb and open with pynwb
    #with pynwb.NWBHDF5IO(file,  mode="r+") as f_io:
    #    ### Now add novel data using pynwb in a way thats a lot less brute force, and way more nwb friendly
    #    f = f_io.read()
    #    NWB_f = f
    #    if False:
    #        ##Currently not working ##Class is compiled but attributes are not written to files
    #        meta_class = build_settings(n_metadata)
    #        test = get_class('MetaData', 'NHP')

    #        nwb_meta = test(name='meta', experiment_id=int(12), test='sfesfesf')
    #        NWB_f.add_lab_meta_data(nwb_meta)
    #    else:
    #        ## For now just dump into scratch ##Goes against NWB Conventions however
    #        for key, value in n_metadata.items():
    #            if isinstance(value, dict):
    #                cont = dict_to_list(value)
    #                for x in cont:
    #                   NWB_f.add_scratch([x[1]], name=str(x[0]), notes=str(key))
    #            else:
    #                NWB_f.add_scratch([value], name=str(key), notes="null")
                    
       
        
    #    f_io.write(NWB_f)
        



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
    NHPPath = "C:\\Users\\SMest\\Documents\\NHP_MARM\\210227_Marmoset"
    meta_data = pd.read_csv("C:\\Users\\SMest\\Documents\\NHP_MARM\\210227_Marmoset\\Manual_entry_data.csv", index_col=0)
    protocol = []
    cell_list = meta_data.index.values
    
    for r, celldir, f in os.walk(NHPPath):
              
              for c in celldir: ##Walks through each folder (cell folder) in the root folder

                   c = os.path.join(r, c) ##loads the subdirectory path
                   shutil.copy("C:\\Users\\SMest\\Documents\\NHP\\default.json",c)
              for file in f:
                  if '.nwb' in file:
                   try:
                       cell_name = file.split('.')[0]
                       cell_meta = {'general': meta_data.loc[cell_name].to_dict()}
                       file_path = os.path.join(r,file)
                       print(f"Converting {cell_name}")
                       confirm_metadata(file_path, cell_meta)
                   except:
                     print('fail')


if __name__ == "__main__":
    main()
