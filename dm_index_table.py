import illustris_python as il
import numpy as np
import h5py
import numba as nb
from numba import jit, njit
import tracerFuncs as tF
import funcs
import sys

run = int(sys.argv[1])
start_snap = int(sys.argv[2])
target_snap = int(sys.argv[3])


basePath=f'/virgotng/universe/IllustrisTNG/TNG50-{run}/output'

tF.traceBack_DM(basePath, start_snap, target_snap)