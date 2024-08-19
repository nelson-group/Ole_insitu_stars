import illustris_python as il
import numpy as np
import h5py
import tracerFuncs as tF
import funcs
import sys

run = int(sys.argv[1])
stype = str(sys.argv[2])
target_snap = int(sys.argv[3])

#define function that saves results from TraceAllStars now for every single snapshot
def TraceBackAllInsituStars(basePath,start_snap, target_snap):
    #load all star ids from a specific galaxy
    star_ids = il.snapshot.loadSubset(basePath,start_snap,'stars',fields=['ParticleIDs'])

    #determine all stars that were formed insitu
    insitu = funcs.is_insitu(basePath,start_snap)
    insitu = np.asarray(insitu == 1)
    insitu_star_indices = np.nonzero(insitu)[0]
    
    insituStarsInSubOffset = tF.insituStarsInSubOffset(basePath, start_snap)
    
    sim = basePath[32:39]
    

    #specify path to save results
    result = h5py.File('/vera/ptmp/gc/olwitt/insitu/' + sim + f'/parent_indices_{target_snap}.hdf5','a') 
    
    print('initial loading complete',flush=True)
    
    parent_indices, numTracersInParents = tF.TraceAllStars(basePath,star_ids[insitu_star_indices],\
                                                   start_snap,target_snap,insituStarsInSubOffset)  
    #save results in hdf5 file

    grp = result.create_group(f'snap_{target_snap}')
    grp.create_dataset("parent_indices", data=parent_indices)
    grp.create_dataset('numTracersInParents',data=numTracersInParents)
    print(target_snap, 'done',end = '; ', flush=True)
    result.close()
    return

def TraceBackAllExsituStars(basePath,start_snap, target_snap):
    #load all star ids from a specific galaxy
    star_ids = il.snapshot.loadSubset(basePath, start_snap, 'stars', fields = ['ParticleIDs'])

    #determine all stars that were formed ex-situ
    exsitu = funcs.is_insitu(basePath,start_snap)
    exsitu = np.asarray(exsitu == 0) #funcs.is_insitu returns array with 1s (in-situ), 0s (ex-situ) and -1s (not part of a subhalo in snap)
    exsitu_star_indices = np.nonzero(exsitu)[0]
    
    exsituStarsInSubOffset = tF.exsituStarsInSubOffset(basePath, start_snap)
    
    sim = basePath[32:39]
    
    #specify path to save results
    result = h5py.File('/vera/ptmp/gc/olwitt/exsitu/' + sim + f'/parent_indices_{target_snap}.hdf5','a') 
    
    print('initial loading complete',flush=True)
    
    parent_indices, numTracersInParents = tF.TraceAllStars(basePath,star_ids[exsitu_star_indices],\
                                                   start_snap,target_snap,exsituStarsInSubOffset)  
    #save results in hdf5 file

    grp = result.create_group(f'snap_{target_snap}')
    grp.create_dataset("parent_indices", data = parent_indices)
    grp.create_dataset('numTracersInParents', data = numTracersInParents)
    print(target_snap, 'done',end = '; ', flush=True)
    result.close()
    return

basePath=f'/virgotng/universe/IllustrisTNG/TNG50-{run}/output'
if stype == 'insitu':
    TraceBackAllInsituStars(basePath, 99, target_snap)
elif stype == 'exsitu':
    TraceBackAllExsituStars(basePath, 99, target_snap)
else:
    raise Exception('Specify valid star type! (insitu or exsitu)')