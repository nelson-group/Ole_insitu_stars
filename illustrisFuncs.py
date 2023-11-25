import time
import illustris_python as il
import numpy as np
import h5py
import numba as nb
from numba import jit, njit
import tracerFuncs as tF
import locatingFuncs as lF
import funcs
from os.path import isfile
import sys


def group_masses_for_sub_ids(groupMasses, subMasses, groupFirstSub, subIDs):
    all_group_masses = np.zeros(subIDs.shape[0])
    
    #counter starts at first central before the first subID
    halo_counter = int(groupFirstSub[np.max(np.where(groupFirstSub <= subIDs[0])[0])])
    for i in range(subIDs.shape[0]):
        if subIDs[i] == groupFirstSub[halo_counter]:
            all_group_masses[i] = groupMasses[halo_counter]
            halo_counter += 1
            continue
        all_group_masses[i] = subMasses[subIDs[i]]
    
    return all_group_masses

def give_z_array(basePath):
    z = np.zeros(100)
    for i in range(99,-1,-1):
        z[99-i] = il.groupcat.loadHeader(basePath,i)['Redshift']
    return z