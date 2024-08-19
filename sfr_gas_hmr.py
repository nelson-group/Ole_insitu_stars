import time
import illustris_python as il
import numpy as np
import h5py
import numba as nb
from numba import jit, njit
import funcs
from os.path import isfile, isdir
import sys


@jit(nopython = True, parallel = True)
def compute_hmr(gas_masses, gas_pos, sub_pos, gasInSubOffset, numGasInSub, sf_flag, boxSize, frac, use_percentile):
    hmr = np.zeros(sub_pos.shape[0])    
    subhaloFlag = np.ones(sub_pos.shape[0], dtype = np.ubyte)
    
    for i in nb.prange(sub_pos.shape[0]):
        indices_of_sub = np.arange(gasInSubOffset[i], gasInSubOffset[i] + numGasInSub[i])
        
        #only choose star forming gas cells
        indices_of_sub = indices_of_sub[np.nonzero(sf_flag[indices_of_sub])]
        
        if indices_of_sub.shape[0] == 0:
            subhaloFlag[i] = 0
            continue
        
        sub_gas_masses = gas_masses[indices_of_sub]
        sub_gas_pos = gas_pos[indices_of_sub,:]
        
        sub_gas_dist = funcs.dist_vector_nb(sub_pos[i], sub_gas_pos, boxSize)
        
        if use_percentile:
            hmr[i] = np.nanquantile(sub_gas_dist, frac)
            continue
        
        sorted_indices = np.argsort(sub_gas_dist)
        m_tot = np.sum(sub_gas_masses)
        
        tmp_mass = 0
        
        cumsum = np.cumsum(sub_gas_masses[sorted_indices])
        half_index = np.min(np.where(cumsum >= m_tot * frac)[0])
        hmr[i] = sub_gas_dist[sorted_indices[half_index]]
            
    return hmr, subhaloFlag


def SubhaloHalfmassRadGasSfr(basePath, snap, frac, use_percentile):
    """Computes the radius of a sphere for each subhalo in which half of the total mass of the star-forming gas is enclosed."""
    
    start = time.time()
    
    assert frac <= 1 and frac > 0, 'Choose a fraction of the total mass between 0 and one!'
    
    h = il.groupcat.loadHeader(basePath, snap)
    boxSize = h['BoxSize']
    
    gas = il.snapshot.loadSubset(basePath, snap, 'gas', fields = ['Masses','StarFormationRate','Coordinates'])
    gas_masses = gas['Masses'][:]
    gas_pos = gas['Coordinates'][:,:]
    num_gas = gas_masses.shape[0]
    sf = np.nonzero(gas['StarFormationRate'])[0]
    sf_flag = np.zeros(num_gas, dtype = np.ubyte)
    sf_flag[sf] = 1
    del gas, sf
    
    sub_pos = il.groupcat.loadSubhalos(basePath, snap, fields = ['SubhaloPos'])
    
    
    if isinstance(sub_pos, dict):
        print(f'Computation failed! No subhalos at snapshot {target_snap}!')
        return
    
    numGasInSub = il.groupcat.loadSubhalos(basePath, snap, fields = ['SubhaloLenType'])[:,0]
    
    g = h5py.File(basePath[:-6] + 'postprocessing/offsets/offsets_' + str(snap).zfill(3) + '.hdf5','r')
    gasInSubOffset = g['Subhalo/SnapByType'][:,0]
    g.close()
    
    end_loading = time.time()
    print('Loading took ',np.round(end_loading - start,3),' seconds.')
    
    subhaloHalfmassRadGasSfr, subhaloFlag_sfr = compute_hmr(gas_masses, gas_pos, sub_pos, gasInSubOffset, numGasInSub, sf_flag, boxSize,\
                                                            frac, use_percentile)
    
    end_calc = time.time()
    print('Computing took ',np.round(end_calc - end_loading,3),' seconds.')
    
    file = '/vera/ptmp/gc/olwitt/auxCats/' + basePath[32:39] + f'/SubhaloHalfmassRad_Gas_Sfr_{target_snap}_frac{frac}.hdf5'
    if use_percentile:
        file = file[:-5] + '_q' + file[-5:]
    f = h5py.File(file,'w')
    f.create_dataset('subhaloFlag', data = subhaloFlag_sfr)
    f.create_dataset('SubhaloHalfmassRad_Gas_Sfr', data = subhaloHalfmassRadGasSfr)
    f.close()
    
    return subhaloHalfmassRadGasSfr

run = int(sys.argv[1])
target_snap = int(sys.argv[2])

# frac = 0.5 for half-mass radius
frac = float(sys.argv[3])

# use_percentile = True for using the percentile instead of the half-mass radius
# e.g. the median of all gas cell distances instead of the distance of the gas cell defined
# b the radius enclosing half of the total mass of all star-forming gas cells
# type 1 for True, 0 for False
use_percentile = (int(sys.argv[4]) == 1)
basePath = f'/virgotng/universe/IllustrisTNG/TNG50-{run}/output'
SubhaloHalfmassRadGasSfr(basePath, target_snap, frac, use_percentile)