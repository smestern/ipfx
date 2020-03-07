import traceback
from multiprocessing import Pool
from functools import partial
import os
import json
import h5py
from ipfx.stimulus import StimulusOntology
import allensdk.core.json_utilities as ju

import ipfx.feature_vectors as fv
import ipfx.bin.lims_queries as lq
import ipfx.stim_features as stf
import ipfx.stimulus_protocol_analysis as spa
import ipfx.data_set_features as dsf
import ipfx.time_series_utils as tsu
import ipfx.error as er

from ipfx.aibs_data_set import AibsDataSet

from allensdk.core.cell_types_cache import CellTypesCache as ctc
from allensdk.api.queries.biophysical_api import *
import allensdk.model.biophysical.run_simulate as bi
import LFPy



def main():
    working_dir = 'C:\\Users\\SMest\\source\\repos\\smestern\\ipfx\\biomodel\\'
    bp = BiophysicalApi()
    cache = ctc()
    cells = cache.get_cells()
    for x in cells:
        bmodel = False
        
        try:
            nmod = bp.get_neuronal_models(x['id'])
            bmodel = True
        except:
            bmodel = False
        if nmod != []:
            print(x['id'])
            for i in nmod:
                
                data = bp.cache_data(i['id'], working_dir)
                with open(working_dir + 'manifest.json', "r") as f:
                        json_data = json.load(f)
            
                print('d')
        #os.remove(working_dir)
              
                
                

if __name__ == "__main__":
    main()
