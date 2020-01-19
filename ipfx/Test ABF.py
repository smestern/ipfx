import pandas as pd
import numpy as np
import logging
import pyabf
from ipfx.sweep import Sweep,SweepSet

import abf_dataset

abf = 'C:\\Users\\SMest\\Pictures\\Cluster-Images\\2019_12_13_0082.abf'

abf_set = abf_dataset.ABFDataSet(abf_file=abf)



print(abf_set)
