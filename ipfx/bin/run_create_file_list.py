import os
import glob
import logging
import numpy as np

FILE_PATH = "/media/smestern/Expansion/NHP_MARM_2"

NWBS = list(glob.glob(FILE_PATH + "/**/*.nwb", recursive=True))

np.savetxt(FILE_PATH+"nwb_paths.txt", NWBS, fmt="%s")