import illustris_python as il
import numpy as np
import h5py
import numba as nb
from numba import jit, njit
import tracerFuncs as tF
import funcs
import sys

target_snap = int(sys.argv[1])
#define function that saves results from TraceAllStars now for every single snapshot
def TraceBackAllInsituStars_allSnaps(basePath,start_snap, target_snap):
    #load all star ids from a specific galaxy
    star_ids = il.snapshot.loadSubset(basePath,start_snap,'stars',fields=['ParticleIDs'])

    #determine all stars that were formed insitu
    insitu = funcs.is_insitu(basePath,start_snap)
    insitu = np.asarray(insitu == 1)
    insitu_star_indices = np.nonzero(insitu)[0]
    
    insituStarsInSubOffset = tF.insituStarsInSubOffset(basePath, start_snap)
    
    sim = basePath[32:39]
    
    #result = h5py.File('files/' + sim + '/all_parent_indices.hdf5','w')
    result = h5py.File('/vera/ptmp/gc/olwitt/' + sim + f'/parent_indices_{target_snap}.hdf5','a')    
    
    #run function for every snapshot
    for _ in range(1):
        parent_indices, numTracersInParents = tF.TraceAllStars(basePath,star_ids[insitu_star_indices],\
                                                       start_snap,target_snap,insituStarsInSubOffset)  
        #save results in hdf5 file

        grp = result.create_group(f'snap_0{target_snap}')
        dset = grp.create_dataset("parent_indices", parent_indices.shape, dtype=float)
        dset[:] = parent_indices
        dset2 = grp.create_dataset('numTracersInParents',numTracersInParents.shape, dtype=float)
        dset2[:] = numTracersInParents
        print(target_snap, 'done',end = '; ', flush=True)
    result.close()
    return

basePath='/virgotng/universe/IllustrisTNG/TNG50-1/output'
TraceBackAllInsituStars_allSnaps(basePath, 99, target_snap)