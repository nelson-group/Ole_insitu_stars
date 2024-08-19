import sys
sys.path.append('/vera/u/olwitt/illustris_python/illustris_python')
from loadMPBs import loadMPBs

import time
import illustris_python as il
import matplotlib.pyplot as plt
import numpy as np
import h5py
import numba as nb
from numba import jit, njit
import tracerFuncs as tF
import locatingFuncs as lF

import funcs

run = int(sys.argv[1])
stype = str(sys.argv[2])
if stype.lower() not in ['insitu', 'exsitu', 'in-situ', 'ex-situ', 'in', 'ex']:
    raise Exception( 'Specify valid star type!')
    
    
basePath=f'/virgotng/universe/IllustrisTNG/TNG50-{run}/output'
lF.save_location(basePath, stype)