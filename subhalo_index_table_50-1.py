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
import plotFuncs as pF

import funcs

def save_location(basePath, start_snap = 99):
    start = time.time()
    h_const = il.groupcat.loadHeader(basePath, start_snap)['HubbleParam']

    num_subs = il.groupcat.loadSubhalos(basePath, start_snap, fields = ['SubhaloMass']).shape[0]
    sub_ids = np.arange(num_subs)
    
    #load all MPBs
    tree_ids = loadMPBs(basePath,start_snap,ids = sub_ids, fields=['SubfindID'])
    
    snap = np.arange(13,1,-1)
    
    n = snap.size

    #necessary offsets, when not every tracer is important:
    insituStarsInSubOffset = tF.insituStarsInSubOffset(basePath, start_snap)
    f = h5py.File('/vera/ptmp/gc/olwitt/' + basePath[32:39] + '/parent_indices_99.hdf5','r')
    parent_indices = f['snap_099/parent_indices'][:,0]
    num_tracers = parent_indices.shape[0]
    del parent_indices
    
    #check, whether old variable names are used
    if f.__contains__('snap_099/numTracersInParents'):
        numTracersInParents = f['snap_099/numTracersInParents'][:]
    else:
        numTracersInParents = f['snap_099/tracers_in_parents_offset'][:]
    f.close()
    
    parentsInSubOffset = tF.tracersInSubhalo(insituStarsInSubOffset,numTracersInParents).astype(int)
    parentsInSubOffset  = np.insert(parentsInSubOffset, 0, 0)
    
    tot_time = 0
    tot_time_locating = 0
    tot_time_isInMP = 0
    
    save_file = h5py.File('/vera/ptmp/gc/olwitt/' + basePath[32:39] + '/subhalo_index_table.hdf5','a')
    
    del save_file['snap_013']
    
    for i in range(n):
        start_loop = time.time()
        
        f = h5py.File('/vera/ptmp/gc/olwitt/' + basePath[32:39] + f'/parent_indices_{snap[i]}.hdf5','r')
        parent_indices = f[f'snap_0{snap[i]}/parent_indices'][:,:]       
        f.close()    
        if snap[i] < 10:
            str_snap = f'0{snap[i]}'
        else:
            str_snap = str(snap[i])
        
        g = h5py.File(basePath[:-6] + 'postprocessing/offsets/offsets_0' + str_snap + '.hdf5','r')
        starsInSubOffset_target_snap = g['Subhalo/SnapByType'][:,4]
        gasInSubOffset_target_snap = g['Subhalo/SnapByType'][:,0]
        g.close()
        
        numInSub_target_snap = il.groupcat.loadSubhalos(basePath, snap[i],fields=['SubhaloLenType'])
        
        grp = save_file.create_group('snap_0'+str_snap)
        
        #run function to determine location of parents
        location, isInMP, _ , time_locating, time_isInMP=\
        location_of_parents(basePath, start_snap = 99, target_snap = snap[i], tree_ids = tree_ids,\
                            parent_indices = parent_indices, help_offsets = parentsInSubOffset, sub_ids = sub_ids,\
                            starsInSubOffset_target_snap = starsInSubOffset_target_snap,\
                            gasInSubOffset_target_snap = gasInSubOffset_target_snap,\
                           numInSub_target_snap = numInSub_target_snap, random_frac = 1)
        
        ds = grp.create_dataset('location', data = location)
        ds2 = grp.create_dataset('isInMP', data = isInMP)
            
        end_loop = time.time()
        tot_time += (end_loop - start_loop)
        tot_time_locating += time_locating
        tot_time_isInMP += time_isInMP
        
        print(snap[i], 'done;',end = ' ')
    print('\n average total time per snapshot: ', tot_time/n)
    print('average time for locating per snapshot: ', tot_time_locating/n)
    print('average time for checking MPB per snapshot: ', tot_time_isInMP/n)
    print('total time: ', end_loop - start)
    save_file.close()    
    
    return

def location_of_parents(basePath, start_snap, target_snap, tree_ids,\
                        parent_indices, help_offsets, sub_ids, starsInSubOffset_target_snap, gasInSubOffset_target_snap,\
                        numInSub_target_snap, random_frac = 1):
    """first output returns the subhalo index if the tracer parent particle sits in a galaxy or -1 if it's in the IGM
    second output states, whether parent particle is inside main progenitor at target snapshot"""
    
    start = time.time()
    
    assert random_frac > 0 and random_frac <= 1, 'random fraction has to be > 0 and <= 1!'
    #load number of particles per galaxy to avoid classifying parents as bound to a galaxy 
    #while they're in reality "inner fuzz" of a halo
    
    gasNumInSub_target_snap = numInSub_target_snap[:,0].copy()
    starNumInSub_target_snap = numInSub_target_snap[:,4].copy()
    del numInSub_target_snap
    
    #for resoltution comparison reasons: only use <random_frac> fraction of all tracers:
    if random_frac < 1:
        rng = np.random.default_rng()
        random_parent_indices = np.zeros(parent_indices.shape)
        new_help_offsets = np.zeros(help_offsets.shape[0]).astype(int)
        for i in range(0, help_offsets.shape[0] - 1):
            indices = np.arange(help_offsets[i],help_offsets[i+1])
            size = int(indices.size * random_frac)
            new_help_offsets[i+1] = size + new_help_offsets[i]
            if size > 0:
                parent_indices_indices = rng.choice(indices, size, replace = False, shuffle = False).astype(int)
                random_parent_indices[new_help_offsets[i]:new_help_offsets[i+1]] =\
                parent_indices[np.sort(parent_indices_indices)]
    
        help_offsets = new_help_offsets.copy()
        not_zero = np.where(random_parent_indices[:,0] != 0)[0]
        random_parent_indices = random_parent_indices[not_zero,:]
        parent_indices = random_parent_indices.copy()

        assert parent_indices.shape[0] == new_help_offsets[-1]
        assert new_help_offsets[0] == 0
        
        del random_parent_indices, new_help_offsets
        
    #find parent index in offset files in NlogM
    location = pF.searchsorted_gas_stars(parent_indices, gasInSubOffset_target_snap, gasNumInSub_target_snap,\
                                               starsInSubOffset_target_snap, starNumInSub_target_snap)
    
    time_locating = time.time()
    
#     print(help_offsets[10481:10484])
#     print(location[help_offsets[10482]:help_offsets[10483]])
#     print(np.where(location[help_offsets[10482]:help_offsets[10483]] == -1)[0].size)
#     print(np.where(location == -1)[0][:1000])
#     print(parent_indices[np.where(location == -1)[0],0][:1000])
    
    #now identify parents that are still in their (main progenitor) galaxy
    
    isInMP = np.empty(parent_indices.shape[0],dtype = bool)
    isInMP.fill(False)
    tree_check = list(tree_ids)
    
    #determine missing trees:
    missing = []
    counter = 0
    
    for i in nb.prange(sub_ids[-1]):
        if i != tree_check[counter]:
            missing.append(i)
            i += 1
            continue
        counter += 1
    
    test = 0
    for j in nb.prange(0,help_offsets.shape[0] - 1): #loop over all relevant galaxies at z=0
        #find all associated particles:
        parentIndicesInSub = np.arange(help_offsets[j],help_offsets[j + 1]).astype(int)
        
        if j in missing or parentIndicesInSub.size == 0: #if subhalo hast no tree, skip it and assign "False"
            if j in missing:
                test += parentIndicesInSub.size
            continue
        if tree_ids[sub_ids[j]]['SubfindID'].shape[0] <= start_snap - target_snap: #if tree doesn't reach until target_snap
            test += parentIndicesInSub.size            
            continue
            
        main_prog = tree_ids[sub_ids[j]]['SubfindID'][start_snap - target_snap]
        where = np.where(location[parentIndicesInSub] == main_prog)[0] + parentIndicesInSub[0]
        isInMP[where] = True

        #print('main progenitor: ', main_prog)
        #print(location[parentIndicesInSub][:10],location[parentIndicesInSub][-10:])
        
        if target_snap == 99:
            assert (np.where(location[parentIndicesInSub] == main_prog)[0].shape[0]) == parentIndicesInSub.shape[0],\
            'offsets wrong probably'   
            assert isInMP[parentIndicesInSub].all() == True, 'offsets wrong probably'
    
    time_isInMP = time.time()
    
    return location, isInMP, help_offsets, time_locating - start, time_isInMP - start

basePath='/virgotng/universe/IllustrisTNG/TNG50-1/output'
save_location(basePath)