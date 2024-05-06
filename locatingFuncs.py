import numpy as np
import h5py
from numba import jit, njit
import numba as nb
import illustris_python as il
import tracerFuncs as tF
import funcs
import locatingFuncs as lF
import time
from os.path import isfile, isdir
import os

import sys
sys.path.append('/vera/u/olwitt/illustris_python/illustris_python')
from loadMPBs import loadMPBs

def location_of_parents(basePath, start_snap, target_snap, tree_ids,\
                        parent_indices, help_offsets, sub_ids, starsInSubOffset_target_snap, gasInSubOffset_target_snap,\
                        numInSub_target_snap, GFS_target_snap, random_frac = 1):
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
        
    #find parent index in offset files in NlogM (for ALL subhalos)
    location = lF.searchsorted_gas_stars(parent_indices, gasInSubOffset_target_snap, gasNumInSub_target_snap,\
                                               starsInSubOffset_target_snap, starNumInSub_target_snap)
    
    time_locating = time.time()
    
    #now identify parents that are still in their (main progenitor) galaxy or in one of its satellites
    
    isInMP = np.full(parent_indices.shape[0], 0, dtype = int)
    isInMP_sat = np.full(parent_indices.shape[0], 0, dtype = int)
    tree_check = list(tree_ids)
    
    #determine missing trees:
    missing = []
    counter = 0
    
    for i in range(sub_ids[-1]):
        if i != tree_check[counter]:
            missing.append(i)
            i += 1
            continue
        counter += 1
    
    test = 0
    max_central_id = np.max(GFS_target_snap)
    for j in range(0,help_offsets.shape[0] - 1): #loop over all relevant galaxies at z=0
        #find all associated particles:
        parentIndicesInSub = np.arange(help_offsets[j], help_offsets[j + 1]).astype(int)
        
        if j in missing or parentIndicesInSub.size == 0: #if subhalo hast no tree, skip it and assign "False"
            if j in missing:
                test += parentIndicesInSub.size
            continue
        if tree_ids[sub_ids[j]]['SubfindID'].shape[0] <= start_snap - target_snap: #if tree doesn't reach until target_snap
            test += parentIndicesInSub.size            
            continue
            
        main_prog = tree_ids[sub_ids[j]]['SubfindID'][start_snap - target_snap]
        main_prog_central_index = np.where(GFS_target_snap == main_prog)[0]
        where_mp = np.where(location[parentIndicesInSub] == main_prog)[0] + parentIndicesInSub[0]
        isInMP[where_mp] = 1
        
        if main_prog_central_index.size > 0:
            if main_prog == max_central_id:
                next_central = gasNumInSub_target_snap.shape[0]
            else:
                next_central = GFS_target_snap[main_prog_central_index + 1]
            where_mp_sat = np.where(np.logical_and(location[parentIndicesInSub] > main_prog,\
                                                   location[parentIndicesInSub] < next_central))[0] + parentIndicesInSub[0]
            isInMP_sat[where_mp_sat] = 1
        #print('main progenitor: ', main_prog)
        #print(location[parentIndicesInSub][:10],location[parentIndicesInSub][-10:])
        
        if target_snap == 99:
            assert (np.where(location[parentIndicesInSub] == main_prog)[0].shape[0]) == parentIndicesInSub.shape[0],\
            'offsets wrong probably'   
            assert np.all(isInMP[parentIndicesInSub] == 1), 'offsets wrong probably'
    
#     if target_snap == 99:
#         print(test)
    
    time_isInMP = time.time()
    return location, isInMP, isInMP_sat, help_offsets, time_locating - start, time_isInMP - time_locating

def save_location(basePath, stype, start_snap = 99):
    start = time.time()
    h_const = il.groupcat.loadHeader(basePath, start_snap)['HubbleParam']

    num_subs = il.groupcat.loadSubhalos(basePath, start_snap, fields = ['SubhaloMass']).shape[0]
    sub_ids = np.arange(num_subs)
    
    #load all MPBs
    tree_ids = loadMPBs(basePath, start_snap, ids = sub_ids, fields=['SubfindID'])

    #necessary offsets, when not every tracer is important:
    if stype == 'insitu':
        insituStarsInSubOffset = tF.insituStarsInSubOffset(basePath, start_snap)
    elif stype == 'exsitu':
        insituStarsInSubOffset = tF.exsituStarsInSubOffset(basePath, start_snap)
    
    file = '/vera/ptmp/gc/olwitt/' + stype + '/' + basePath[32:39] + f'/parent_indices_{start_snap}.hdf5'
    if not isfile(file):
        file = 'files/' + basePath[32:39] + '/all_parent_indices.hdf5'
    f = h5py.File(file,'r')  
    parent_indices = f[f'snap_{start_snap}/parent_indices'][:,0]
    num_tracers = parent_indices.shape[0]
    del parent_indices
    
    #check, whether old variable names are used
    if f.__contains__(f'snap_{start_snap}/numTracersInParents'):
        numTracersInParents = f[f'snap_{start_snap}/numTracersInParents'][:]
    else:
        numTracersInParents = f[f'snap_{start_snap}/tracers_in_parents_offset'][:]
        
    #check lowest saved parent index table snapshot
    min_snap = funcs.find_file_with_lowest_number('/vera/ptmp/gc/olwitt/' + stype + '/' + basePath[32:39])
    snaps = np.arange(99,min_snap - 1,-1)
    
    n = snaps.size    
    
    f.close()
    
    parentsInSubOffset = tF.tracersInSubhalo(insituStarsInSubOffset,numTracersInParents).astype(int)
    parentsInSubOffset  = np.insert(parentsInSubOffset, 0, 0)
    
    tot_time = 0
    tot_time_locating = 0
    tot_time_isInMP = 0
    
    save_file = h5py.File('/vera/ptmp/gc/olwitt/' + stype + '/' + basePath[32:39] + '/subhalo_index_table.hdf5','w')
    
    for i in range(n):
        start_loop = time.time()
        
        file = '/vera/ptmp/gc/olwitt/' + stype + '/' + basePath[32:39] + f'/parent_indices_{snaps[i]}.hdf5'
        if not isfile(file):
            file = 'files/' + basePath[32:39] + '/all_parent_indices.hdf5'
        f = h5py.File(file,'r')
        parent_indices = f[f'snap_{snaps[i]}/parent_indices'][:,:]       
        f.close()    
        
        g = h5py.File(basePath[:-6] + 'postprocessing/offsets/offsets_' + str(snaps[i]).zfill(3) + '.hdf5','r')
        if 'Subhalo' in list(g.keys()): # equivalent to g.__contains__('Subhalo')
            starsInSubOffset_target_snap = g['Subhalo/SnapByType'][:,4]
            gasInSubOffset_target_snap = g['Subhalo/SnapByType'][:,0]
        else:
            g.close()
            continue
        g.close()
        
        numInSub_target_snap = il.groupcat.loadSubhalos(basePath, snaps[i],fields=['SubhaloLenType'])
        GFS = il.groupcat.loadHalos(basePath, snaps[i], fields = ['GroupFirstSub'])
        
        grp = save_file.create_group('snap_'+ str(snaps[i]))
        
        #run function to determine location of parents
        location, isInMP, isInMP_sat, _ , time_locating, time_isInMP = \
        lF.location_of_parents(basePath, start_snap = 99, target_snap = snaps[i], tree_ids = tree_ids,\
                            parent_indices = parent_indices, help_offsets = parentsInSubOffset, sub_ids = sub_ids,\
                            starsInSubOffset_target_snap = starsInSubOffset_target_snap,\
                            gasInSubOffset_target_snap = gasInSubOffset_target_snap,\
                           numInSub_target_snap = numInSub_target_snap, GFS_target_snap = GFS, random_frac = 1)
        
        grp.create_dataset('location', data = location)
#         grp.create_dataset('isInMP', data = isInMP)
#         grp.create_dataset('isInMP_satellite', data = isInMP_sat)
            
        isInCentral = np.full(location.shape[0], 0, dtype = int)

#         _,_,central_indices = np.intersect1d(GFS[np.where(GFS != -1)],location, return_indices = True)
        central_indices = np.nonzero(np.isin(location, GFS[np.where(GFS != -1)]))[0]
        isInCentral[central_indices] = 1 #mark which parents are located within a central galaxy with 1 and ...
        
#         grp.create_dataset('isInCentral', data = isInCentral)
        
        result = np.zeros(location.shape[0], dtype = np.ubyte)
        result[np.where(np.logical_and(np.logical_and(isInMP == 0, isInCentral == 0), isInMP_sat == 0))] = 0
        result[np.where(np.logical_and(np.logical_and(isInMP == 1, isInCentral == 0), isInMP_sat == 0))] = 1
        result[np.where(np.logical_and(np.logical_and(isInMP == 1, isInCentral == 1), isInMP_sat == 0))] = 2
        
        result[np.where(np.logical_and(np.logical_and(isInMP == 1, isInCentral == 0), isInMP_sat == 1))] = 3 # should not exist -> check
        assert np.where(np.logical_and(np.logical_and(isInMP == 1, isInCentral == 0), isInMP_sat == 1))[0].shape[0] == 0
        
        result[np.where(np.logical_and(np.logical_and(isInMP == 1, isInCentral == 1), isInMP_sat == 1))] = 4 # should not exist
        assert np.where(np.logical_and(np.logical_and(isInMP == 1, isInCentral == 1), isInMP_sat == 1))[0].shape[0] == 0
        
        result[np.where(np.logical_and(np.logical_and(isInMP == 0, isInCentral == 1), isInMP_sat == 0))] = 5
        
        result[np.where(np.logical_and(np.logical_and(isInMP == 0, isInCentral == 1), isInMP_sat == 1))] = 6 #should not exist
        assert np.where(np.logical_and(np.logical_and(isInMP == 0, isInCentral == 1), isInMP_sat == 1))[0].shape[0] == 0
        
        result[np.where(np.logical_and(np.logical_and(isInMP == 0, isInCentral == 0), isInMP_sat == 1))] = 7
        
        grp.create_dataset('location_type', data = result)
        
        del isInCentral, isInMP, isInMP_sat, location, GFS, parent_indices
        
        end_loop = time.time()
        tot_time += (end_loop - start_loop)
        tot_time_locating += time_locating
        tot_time_isInMP += time_isInMP
        
        print(snaps[i], 'done;', end = ' ', flush = True)
    print('\n average total time per snapshot: ', tot_time/n)
    print('average time for locating per snapshot: ', tot_time_locating/n)
    print('average time for checking MPB per snapshot: ', tot_time_isInMP/n)
    print('total time: ', end_loop - start)
    save_file.close()    
    
    return

def fracs_w_mass_bins(basePath, stype, sub_ids, start_snap = 99, random_frac = 1):
    start = time.time()
    h_const = il.groupcat.loadHeader(basePath, start_snap)['HubbleParam']
    
    #check lowest saved parent index table snapshot
    min_snap = funcs.find_file_with_lowest_number('/vera/ptmp/gc/olwitt/' + stype + '/' + basePath[32:39])
    snaps = np.arange(99,min_snap - 1,-1)
    
    n = snaps.size
    
    z = np.zeros(n)

    #necessary offsets, when not every tracer is important:
    if stype == 'insitu':
        insituStarsInSubOffset = tF.insituStarsInSubOffset(basePath, start_snap)
    elif stype == 'exsitu':
        insituStarsInSubOffset = tF.exsituStarsInSubOffset(basePath, start_snap)
    else:
        raise Exception('Invalid star/particle type!')
        
    file = '/vera/ptmp/gc/olwitt/' + stype + '/' + basePath[32:39] + f'/parent_indices_{start_snap}.hdf5'
    f = h5py.File(file,'r')   

    parent_indices = f[f'snap_{start_snap}/parent_indices'][:,0]
    num_tracers = parent_indices.shape[0]
    del parent_indices
    
    #check, whether old variable names are used
    if f.__contains__(f'snap_{start_snap}/numTracersInParents'):
        numTracersInParents = f[f'snap_{start_snap}/numTracersInParents'][:]
    else:
        numTracersInParents = f[f'snap_{start_snap}/tracers_in_parents_offset'][:]  
    
    f.close()
    
    parentsInSubOffset = tF.tracersInSubhalo(insituStarsInSubOffset,numTracersInParents).astype(int)
    parentsInSubOffset  = np.insert(parentsInSubOffset, 0, 0)
    
    help_offsets = np.zeros(sub_ids.shape[0])
    which_indices = np.zeros(num_tracers)
    isGalaxy = np.empty(sub_ids.shape, dtype = bool)
    isGalaxy.fill(False)
    
    before_indices = time.time()
    
    # determine galaxies without tree from isInMP to not include them later on
    location_file = h5py.File('/vera/ptmp/gc/olwitt/' + stype + '/' + basePath[32:39] + '/subhalo_index_table.hdf5','r')
    location_type = location_file[f'snap_{start_snap}/location_type'][:]
    isInMP = np.zeros(location_type.shape[0], dtype = np.ubyte)
    isInMP[np.isin(location_type,np.array([1,2]))] = 1
    del location_type
    location_file.close()
    
    #determine missing trees:
    missing = []
    counter = 0
    max_ind = max(parentsInSubOffset)
    for i in range(sub_ids[-1]):
        index = parentsInSubOffset[i]
        if index == max_ind:
            counter += 1
            continue
        if isInMP[index] == 0:
            missing.append(i)
            i += 1
            continue
        counter += 1
#     print(missing)
    
    counter = 0
    for i in range(1,sub_ids.shape[0] + 1):
        indcs = np.arange(parentsInSubOffset[sub_ids[i-1]],parentsInSubOffset[sub_ids[i-1]+1])
        if indcs.size > 0 and sub_ids[i-1] not in missing:
            isGalaxy[i-1] = True
        which_indices[counter:counter + indcs.shape[0]] = indcs
        help_offsets[i-1] = indcs.shape[0]
        counter += indcs.shape[0]
        
    del indcs, counter, isInMP
    
#     print(sub_ids.shape, len(missing), np.nonzero(isGalaxy)[0].shape)
#     sub_ids = np.delete(sub_ids, np.array(missing))
#     print(sub_ids.shape)
    
    #trim zeros at the end:
    which_indices = np.trim_zeros(which_indices,'b').astype(int)
    
    #compute correct offsets:
    ## states, which indices correspond to which subhalo from sub_ids
    help_offsets = np.cumsum(help_offsets).astype(int)
    help_offsets = np.insert(help_offsets,0,0)
    
    #which_indices = np.arange(num_tracers)
    #help_offsets = parentsInSubOffset
    
            #only take random fraction out of all tracers to compare different resolutions
#         if random_frac < 1:
#             rng = np.random.default_rng()
#             random_parent_indices = np.zeros(parent_indices.shape)
#             new_help_offsets = np.zeros(help_offsets.shape[0]).astype(int)
#             for i in range(0, help_offsets.shape[0] - 1):
#                 indices = np.arange(help_offsets[i],help_offsets[i+1])
#                 size = int(indices.size * random_frac)
#                 new_help_offsets[i+1] = size + new_help_offsets[i]
#                 if size > 0:
#                     parent_indices_indices = rng.choice(indices, size, replace = False, shuffle = False).astype(int)
#                     random_parent_indices[new_help_offsets[i]:new_help_offsets[i+1]] =\
#                     parent_indices[np.sort(parent_indices_indices)]

#             help_offsets = new_help_offsets.copy()
#             not_zero = np.where(random_parent_indices[:,0] != 0)[0]
#             random_parent_indices = random_parent_indices[not_zero,:]
#             parent_indices = random_parent_indices.copy()

#             assert parent_indices.shape[0] == new_help_offsets[-1]
#             assert new_help_offsets[0] == 0
        
#             del random_parent_indices, new_help_offsets
    
#     location_file = h5py.File('files/' + basePath[32:39] + '/subhalo_index_table.hdf5','r')
    location_file = h5py.File('/vera/ptmp/gc/olwitt/' + stype + '/' + basePath[32:39] + '/subhalo_index_table.hdf5','r')
    
    mp = np.zeros(n)
    mp_stars = np.zeros(n)
    mp_gas = np.zeros(n)
    igm = np.zeros(n)
    sub = np.zeros(n)
    other_centrals = np.zeros(n)
    other_satellites = np.zeros(n)
    mp_satellites = np.zeros(n)
    
    total = np.zeros(n)

    nums = np.zeros((n,5,8))
    gal_comp = np.zeros((n,sub_ids.shape[0],8)) #galaxy composition
#     stars_in_gals = np.zeros((n, sub_ids.shape[0]))
    
    #check accretion origins:
    
    # three disjoint modes according to nelson et al. (2013):
    
    # smooth accretion = directly_from_igm
    #                  = fresh accretion from igm + wind_recycled (nep)
    
    # clumpy accretion = mergers
    # stripped/ejected from halos = stripped_from_halos
    
    directly_from_igm = np.ones(which_indices.shape[0], dtype = int) #output
    smooth = np.zeros(which_indices.shape[0], dtype = int) # helper array
    long_range_wind_recycled = smooth.copy() # output
    from_other_halos = smooth.copy() # output
    mergers = smooth.copy() # output
#     back_to_sat = smooth.copy() # helper array
    merger_before_smooth = smooth.copy() # helper array
    
    start_loop = time.time()
    print('time for indices: ',start_loop-before_indices, flush = True)
    print('before loop: ',start_loop-start, flush = True)
    
    for i in range(n): #loop over all snapshots
        file = '/vera/ptmp/gc/olwitt/' + stype + '/' + basePath[32:39] + f'/parent_indices_{snaps[i]}.hdf5'
        f = h5py.File(file,'r')      

        parent_indices = f[f'snap_{snaps[i]}/parent_indices'][:,:]
        f.close()
        if i==1:
            start_loop = time.time()
        
        #only consider indices of relevant galaxies        
        parent_indices = parent_indices[which_indices,:]
        
        if(i==1):
            end_files = time.time()
            print('time for loading snap from files: ',end_files-start_loop, flush = True)
        
        before_locating = time.time()
        
        if not location_file.__contains__(f'snap_{snaps[i]}'):
            continue
        
        #load location of parents from file (but only tracers from subhalos that we are interested in)
        location = location_file[f'snap_{snaps[i]}/location'][:]
        location = location[which_indices]
        
        # decode new datatype:
        location_type = location_file[f'snap_{snaps[i]}/location_type'][:]
        location_type = location_type[which_indices]
        
        isInMP = np.zeros(location.shape[0], dtype = np.ubyte)
        isInCentral = np.zeros(location.shape[0], dtype = np.ubyte)
        isInMP_sat = np.zeros(location.shape[0], dtype = np.ubyte)
        
        isInMP[np.isin(location_type,np.array([1,2]))] = 1 #in theory also where location_type = 3,4 but there are no tracers with 3,4
        isInCentral[np.isin(location_type,np.array([2,5]))] = 1 #in theory also where location_type = 4,6 but there are no tracers with 4,6
        isInMP_sat[np.isin(location_type,np.array([7]))] = 1 #in theory also where location_type = 3,4,6 but there are no tracers with 3,4,6
        
#         isInMP = location_file[f'snap_{snaps[i]}/isInMP'][:]
#         isInMP = isInMP[which_indices]
        
#         isInCentral = location_file[f'snap_{snaps[i]}/isInCentral'][:]
#         isInCentral = isInCentral[which_indices]
        
#         isInMP_sat = location_file[f'snap_{snaps[i]}/isInMP_satellite'][:]
#         isInMP_sat = isInMP_sat[which_indices]
        
        if i==0:
            print(f'{np.where(isInMP == 0)[0].shape[0]} tracers not in MP at z=0')
            print(f'{np.where(location == -1)[0].shape[0]} tracers int the IGM at z=0')
#             assert np.all(isInMP == 1), f'error: {np.where(isInMP == 0)[0].shape[0]} tracers not in MP at z=0!'
        
        if(i==1):
            end_locate = time.time()
            print('total time for locating: ',end_locate-before_locating, flush = True)
        #load baryonic masses
        sub_masses_stars = il.groupcat.loadSubhalos(basePath,snaps[i],fields=['SubhaloMassType'])[:,4] * 1e10/h_const
        sub_masses_gas = il.groupcat.loadSubhalos(basePath,snaps[i],fields=['SubhaloMassType'])[:,0] * 1e10/h_const
        sub_masses = sub_masses_stars + sub_masses_gas #consider baryonic mass, not stellar mass
        
        m_tot = np.sum(sub_masses)
        
        #create mass bins
        mass_bin1 = np.where(np.logical_and(sub_masses != 0, sub_masses < 1e9))[0]
        mass_bin2 = np.where(np.logical_and(sub_masses >= 1e9, sub_masses < 1e10))[0]
        mass_bin3 = np.where(np.logical_and(sub_masses >= 1e10, sub_masses < 1e11))[0]
        mass_bin4 = np.where(np.logical_and(sub_masses >= 1e11, sub_masses < 1e12))[0]
        mass_bin5 = np.where(sub_masses >= 1e12)[0]  
        
        if(i==1):
            end_create_bins = time.time()
            print('total time for creating bins: ',end_create_bins-end_locate, flush = True)
        
        #add numbers to mass bins
        nums[i,:,:], gal_comp[i,:,:] = lF.binning(parent_indices, location, isInMP, isInMP_sat, isInCentral, sub_ids,\
                                                                      help_offsets,mass_bin1, mass_bin2, mass_bin3, mass_bin4, mass_bin5)
        
        star_mask = np.where(parent_indices[:,1] == 1)[0]
        gas_mask = np.where(parent_indices[:,1] == 0)[0]
        
        mp_stars[i] = np.where(isInMP[star_mask] == 1)[0].shape[0] #number of star parents in the MP
        mp_gas[i] = np.where(isInMP[gas_mask] == 1)[0].shape[0] #number of gas parents in the MP
        mp[i] = mp_stars[i] + mp_gas[i] #number of parents in the MP
        
        igm[i] = np.where(location == -1)[0].shape[0] #number of parents in the IGM
        
        assert mp[i] == np.where(isInMP == 1)[0].shape[0], 'MP wrong.'
        
        mp_satellites[i] = np.where(isInMP_sat == 1)[0].shape[0] #number of parents in satellites of the MP
        
        #number of parents in satellites of other central galaxies than the MP
        other_satellites[i] = np.where(np.logical_and(np.logical_and(location != -1, isInCentral == 0),\
                                                      isInMP == 0))[0].shape[0] - mp_satellites[i]
        
        #number of parents in other central galaxies than the MP (other halos)
        other_centrals[i] = np.where(np.logical_and(np.logical_and(location != -1, isInCentral == 1),isInMP == 0))[0].shape[0]
        
        #number of parents in other galaxies than the MP; satellites + other_centrals = sub
        sub[i] = np.where(location != -1)[0].shape[0] - mp[i] #everything not in the igm is in a subhalo (or a FoF halo)
        
        z[i] = il.groupcat.loadHeader(basePath, snaps[i])['Redshift'] 
        total[i] = igm[i] + sub[i] + mp[i]
        
        ##### what fraction of baryonic matter forming in situ stars by z=0 was accreted in a certain way? #####
        if i > 0: #skip start_snapshot
            # mark each parent currently in another galaxy than the MP:
            other_gal_tmp = np.where(np.logical_and(location != -1, isInMP == 0))[0] 
            directly_from_igm[other_gal_tmp] = 0 #assume all came from igm, delete those that were in other galaxies
            
            # smooth accretion (was in IGM, now in MP)
            smooth_tmp = np.where(np.logical_and(np.logical_and(old_isInMP == 1, location == -1), isInMP == 0))[0] 
            smooth[smooth_tmp] = 1 #assume none were smoothly accreted, mark those that are
            
            # mark each parent currently in another halo than the one of the MP (= intergalactic transfer)
            other_halos_tmp = np.where(np.logical_and(np.logical_and(location != -1, isInMP == 0), isInMP_sat == 0))[0] 
            from_other_halos[other_gal_tmp] = 1 #assume none from other halos (or galaxies in general, respectively), mark those that are 
            
            # wind recycling (was in MP, now in IGM, and eventually in MP again (z=0))
            wind_rec_tmp = np.where(np.logical_and(old_location == -1, isInMP == 1))[0] 
            long_range_wind_recycled[wind_rec_tmp] = 1
            
            # mark each parent, that was bound to another subhalo prior to being in the MP
            merger_tmp = np.where(np.logical_and(np.logical_and(isInMP == 0, location != -1), old_isInMP == 1))[0]
            mergers[merger_tmp] = 1
            
            # mark parents, that entered the MP via a merger BEFORE they were accreted smoothly (e.g. due to wind recycling)
            merger_first_tmp = np.where(np.logical_and(smooth == 0, mergers == 1))[0]
            merger_before_smooth[merger_first_tmp] = 1
            
            # mark each parent, that was in the MP but then entered a satellite
#             back_to_sat_tmp = np.where(np.logical_and(isInMP == 1, old_isInMP_sat == 1))[0]
#             back_to_sat[back_to_sat_tmp] = 1
            
        
        old_location = location.copy() #old_location refers to higher snapshot (later time) than location, as snaps[i] decreases in each loop
        old_isInMP = isInMP.copy()
#         old_isInMP_sat = isInMP_sat.copy()
        
        if(i==1):
            end_binning = time.time()
            print('total time for binning: ', end_binning-end_create_bins, flush = True)
            print('total time for first loop: ', end_binning-start_loop, flush = True)
        print(snaps[i], 'done;', end = ' ', flush = True)
    print()
    
    #find all tracers that were in halos at some point AND meet the smooth accretion criterion
    stripped_from_halos = np.zeros(which_indices.shape[0], dtype = int)
    stripped_from_halos_inds = np.where(np.logical_and(smooth == 1, from_other_halos == 1))[0]
    stripped_from_halos[stripped_from_halos_inds] = 1
    
    # corrections:
    
#     print('# false stripped_from_halos tracers: ', np.where(np.logical_and(merger_before_smooth == 1, stripped_from_halos == 1))[0].shape)
    merger_before_wind_rec = np.where(np.logical_and(merger_before_smooth == 1, stripped_from_halos == 1))[0] #false stripped from halos
    stripped_from_halos[np.where(merger_before_smooth == 1)] = 0
#     print('# false mergers tracers: ', np.where(np.logical_and(mergers == 1, stripped_from_halos == 1))[0].shape)
    mergers[np.where(stripped_from_halos == 1)] = 0
    
    #-> smooth (here != nelson13 definition) = directly_from_igm - alwaysInMP + (real) stripped_from_halos
    alwaysInMP = np.where(np.logical_and(smooth == 0, directly_from_igm == 1))[0]
#     print(np.nonzero(smooth)[0].shape[0], np.nonzero(directly_from_igm)[0].shape[0], np.nonzero(stripped_from_halos)[0].shape[0],\
#           np.nonzero(merger_before_wind_rec)[0].shape[0], alwaysInMP.shape[0])
    
#     assert np.nonzero(smooth)[0].shape[0] == np.nonzero(directly_from_igm)[0].shape[0] - alwaysInMP.shape[0] +\
#     np.nonzero(stripped_from_halos)[0].shape[0] + np.nonzero(merger_before_wind_rec)[0].shape[0]
    
    # non-externally-processed (nep) wind recycling
    nep_wind_recycled = np.zeros(which_indices.shape[0], dtype = int)
    nep_wind_recycled_inds = np.where(np.logical_and(directly_from_igm == 1, long_range_wind_recycled == 1))[0]
    nep_wind_recycled[nep_wind_recycled_inds] = 1
    
    del location, old_location, isInMP, old_isInMP, stripped_from_halos_inds, nep_wind_recycled_inds, which_indices
    
    # tracers in multiple categories?
    igm_mergers = np.where(np.logical_and(directly_from_igm == 1, mergers == 1))[0]
    igm_stripped = np.where(np.logical_and(directly_from_igm == 1, stripped_from_halos == 1))[0]
    merger_stripped = np.where(np.logical_and(stripped_from_halos == 1, mergers == 1))[0]
    
#     print('back to satellites: ', np.nonzero(back_to_sat)[0].shape[0])
    
    print('# of tracers in two categories: ',igm_mergers.shape, igm_stripped.shape, merger_stripped.shape)
    print('apparent # of tracers in categories: ', np.nonzero(directly_from_igm)[0].shape[0] + np.nonzero(mergers)[0].shape[0] +\
          np.nonzero(stripped_from_halos)[0].shape[0])
    print('actual total # of tracers: ', mergers.shape)
    no_cat = np.where(np.logical_and(np.logical_and(mergers == 0, directly_from_igm == 0),stripped_from_halos == 0))[0]
    print('# of tracers in no category: ', no_cat.shape)   
    
    print('# no cat from other halos: ', np.where(from_other_halos[no_cat] == 1)[0].shape[0])
    print('# no cat mergers: ', np.where(mergers[no_cat] == 1)[0].shape[0])
    print('# no cat smooth accretion: ', np.where(smooth[no_cat] == 1)[0].shape[0])
#     print(no_cat)
    
    del igm_mergers, igm_stripped, merger_stripped, no_cat, alwaysInMP, smooth#, back_to_sat
    
    # compute accretion channel fractions for every galaxy individually
    gal_accretion_channels = accretion_channels_all_gals(sub_ids, help_offsets, directly_from_igm, from_other_halos, mergers, stripped_from_halos, long_range_wind_recycled, nep_wind_recycled)
    
    # convert arrays into overall fractions (save arrays?)
    directly_from_igm = np.nonzero(directly_from_igm)[0].shape[0] / directly_from_igm.shape[0]
    from_other_halos = np.nonzero(from_other_halos)[0].shape[0] / from_other_halos.shape[0]
    mergers = np.nonzero(mergers)[0].shape[0] / mergers.shape[0]
    stripped_from_halos = np.nonzero(stripped_from_halos)[0].shape[0] / stripped_from_halos.shape[0]
    long_range_wind_recycled = np.nonzero(long_range_wind_recycled)[0].shape[0] / long_range_wind_recycled.shape[0]
    nep_wind_recycled = np.nonzero(nep_wind_recycled)[0].shape[0] / nep_wind_recycled.shape[0]
    
    # only use/save galaxies with at least one tracer
    gal_comp = gal_comp[:,np.nonzero(isGalaxy)[0],:]
    gal_accretion_channels = gal_accretion_channels[np.nonzero(isGalaxy)[0],:]
    location_file.close()
    return mp, mp_stars, mp_gas, igm, sub, other_satellites, mp_satellites, other_centrals, total, nums, z, gal_comp, isGalaxy, directly_from_igm, stripped_from_halos, from_other_halos, mergers, long_range_wind_recycled, nep_wind_recycled, gal_accretion_channels

@njit
def binning(parent_indices, location, isInMP, isInMP_sat, isInCentral, sub_ids, help_offsets, mass_bin1, mass_bin2, mass_bin3, mass_bin4, mass_bin5):
    res = np.zeros((5,8))
    gal_res = np.zeros((help_offsets.shape[0] - 1,8))
#     star_res = np.zeros(help_offsets.shape[0])
    
    #determine mass fractions for every galaxy individually
    for i in range(0,help_offsets.shape[0] - 1):
        indices = np.arange(help_offsets[i],help_offsets[i+1])
        
        star_mask = np.where(parent_indices[indices,1] == 1)[0]
        gas_mask = np.where(parent_indices[indices,1] == 0)[0]
        
#         star_res[i] = star_mask.shape[0] / indices.shape[0] if indices.shape[0] > 0 else 0.
        
        gal_res[i,0] = np.where(isInMP[indices] == 1)[0].shape[0] #number of parents in the MP
        gal_res[i,1] = np.where(isInMP[indices[star_mask]] == 1)[0].shape[0] #number of star parents in the MP
        gal_res[i,2] = np.where(isInMP[indices[gas_mask]] == 1)[0].shape[0] #number of gas parents in the MP
        
        gal_res[i,3] = np.where(location[indices] != -1)[0].shape[0] - gal_res[i,0] # number of parents in other galaxies 
        gal_res[i,5] = np.where(isInMP_sat[indices] == 1)[0].shape[0] #number of parents in satellites of the MP
        gal_res[i,4] = np.where(np.logical_and(np.logical_and(location[indices] != -1, isInCentral[indices] == 0), isInMP[indices] == 0))[0].shape[0] - gal_res[i,5] #number of parents in satellites of other halos
        
        gal_res[i,6] = np.where(np.logical_and(np.logical_and(location[indices] != -1, isInCentral[indices] == 1),isInMP[indices] == 0))[0].shape[0] #number of parents in other centrals (halos)
        
        gal_res[i,7] = indices.shape[0] #total
        gal_res[i,:7] = gal_res[i,:7] / gal_res[i,7] if gal_res[i,7] > 0 else gal_res[i,:7] #obtain mass fractions
    
    #determine mass fractions for entire mass bins
    for i in nb.prange(5):
        mass_bin = mass_bin1 if i==0 else mass_bin2 if i==1 else mass_bin3 if i==2 else mass_bin4 if i==3 else\
        mass_bin5
        indices = np.nonzero(funcs.isin(location,mass_bin))[0]
        star_mask = np.where(parent_indices[indices,1] == 1)[0]
        gas_mask = np.where(parent_indices[indices,1] == 0)[0]
        
        res[i,0] = np.where(isInMP[indices] == 1)[0].shape[0] #number of parents in the MP
        res[i,1] = np.where(isInMP[indices[star_mask]] == 1)[0].shape[0] #number of star parents in the MP
        res[i,2] = np.where(isInMP[indices[gas_mask]] == 1)[0].shape[0] #number of gas parents in the MP
        
        res[i,3] = np.where(location[indices] != -1)[0].shape[0] - res[i,0] # number of parents in other galaxies 
        res[i,5] = np.where(isInMP_sat[indices] == 1)[0].shape[0] #number of parents in satellites of the MP
        res[i,4] = np.where(np.logical_and(np.logical_and(location[indices] != -1, isInCentral[indices] == 0), isInMP[indices] == 0))[0].shape[0] - res[i,5] #number of parents in satellites of other halos
        
        res[i,6] = np.where(np.logical_and(np.logical_and(location[indices] != -1, isInCentral[indices] == 1),isInMP[indices] == 0))[0].shape[0] #number of parents in other centrals (halos)
        
        res[i,7] = indices.shape[0] #total
        res[i,:7] = res[i,:7] / res[i,7] if res[i,7] > 0 else res[i,:7] #obtain mass fractions
    return res, gal_res#, star_res

@jit(nopython = True, parallel = True)
def accretion_channels_all_gals(sub_ids, help_offsets, directly_from_igm, from_other_halos, mergers, stripped_from_halos, long_range_wind_recycled, nep_wind_recycled):
    res = np.zeros((help_offsets.shape[0] - 1,6))
    for i in range(help_offsets.shape[0] - 1):
        indices = np.arange(help_offsets[i],help_offsets[i+1])
        if indices.shape[0] > 0:
            res[i,0] = np.nonzero(directly_from_igm[indices])[0].shape[0] / indices.shape[0]
            res[i,1] = np.nonzero(from_other_halos[indices])[0].shape[0] / indices.shape[0]
            res[i,2] = np.nonzero(mergers[indices])[0].shape[0] / indices.shape[0]
            res[i,3] = np.nonzero(stripped_from_halos[indices])[0].shape[0] / indices.shape[0]
            res[i,4] = np.nonzero(long_range_wind_recycled[indices])[0].shape[0] / indices.shape[0]
            res[i,5] = np.nonzero(nep_wind_recycled[indices])[0].shape[0] / indices.shape[0]
        
    return res


@jit(nopython=True, parallel=True)
def searchsorted_gas_stars(parent_indices, gasInSubOffset_target_snap, gasNumInSub_target_snap, starsInSubOffset_target_snap, starNumInSub_target_snap):#, sub_m_star = None):
    
    location = np.empty(parent_indices.shape[0], np.intp)
    location.fill(-1)
    which = parent_indices[:,1]
    indices = parent_indices[:,0]
    
#     if sub_m_star != None:
#         limit = 10**(9)
#         valid = np.where(sub_m_star > limit)[0]
#     else:
#         valid = np.arange(sub_m_star)
    
    for i in nb.prange(parent_indices.shape[0]):
        if int(which[i]) == 0: #gas parent
            ind = np.searchsorted(gasInSubOffset_target_snap,int(indices[i]),'right')
            ind -= 1 #compensate for how np.searchsorted chooses the index
            
            #mark particles within galaxies below a certain mass limit
#             if ind not in valid:
#                 location[i] = -2
#                 continue
            
            #only important for last galaxy of halo: check whether particle is inner fuzz, i.e. IGM particles
            if gasInSubOffset_target_snap[ind] + gasNumInSub_target_snap[ind] < indices[i]:
                continue
            
        else: #star parent
            ind = np.searchsorted(starsInSubOffset_target_snap,int(indices[i]),'right')
            ind -= 1 #compensate for how np.searchsorted chooses the index
            
#             if ind not in valid:
#                 location[i] = -2
#                 continue
            
            #only important for last galaxy of halo: check whether particle is inner fuzz, i.e. IGM particles
            if starsInSubOffset_target_snap[ind] + starNumInSub_target_snap[ind] < indices[i]:
                continue
                
        location[i]=ind
    return location