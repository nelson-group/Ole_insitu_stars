import numpy as np
import h5py
from numba import jit, njit
import numba as nb
import illustris_python as il
import tracerFuncs as tF
import dm
import funcs
import locatingFuncs as lF
import illustrisFuncs as iF
import time
from os.path import isfile, isdir
import os
import snapshot_mod as sm

import sys
sys.path.append('/vera/u/olwitt/illustris_python/illustris_python')
from loadMPBs import loadMPBs

@jit(nopython = True, parallel = True)
def DM_in_2_shmr(DM_coords, sub_pos, dmInSubOffset, numDMInSubs, shmr, cut, boxSize):
    which_particles = np.zeros(DM_coords.shape[0], dtype = np.ubyte)
    inside_offsets = np.zeros(sub_pos.shape[0], dtype = int)
    
    for i in range(sub_pos.shape[0]):
        sub_indices = np.arange(dmInSubOffset[i],dmInSubOffset[i] + numDMInSubs[i]) #particle indices in subhalo
        sub_distances_dm = funcs.dist_vector_nb(sub_pos[i], DM_coords[sub_indices], boxSize) #particle distances to sub center
        inside = np.where(sub_distances_dm <= cut * shmr[i])[0] #particles inside 2shmr
        which_particles[inside + dmInSubOffset[i]] = 1 #mark those particles
        inside_offsets[i] = inside.shape[0] #save number for offsets
    
    inside_offsets = np.cumsum(inside_offsets)
    inside_offsets = np.insert(inside_offsets,0,0)
    return which_particles, inside_offsets

def traceBack_DM(basePath, start_snap, target_snap):
    start = time.time()
    h = il.groupcat.loadHeader(basePath, start_snap)
    boxSize = h['BoxSize'] #ckpc
    h_const = h['HubbleParam']
    
    
    #load all DM particles
    DM_coords = il.snapshot.loadSubset(basePath, start_snap, 1, fields = ['Coordinates']) #ckpc/h
    
    #load subhalo positions
    sub_pos = il.groupcat.loadSubhalos(basePath, start_snap, fields = ['SubhaloPos']) #ckpc/h
    
    # load DM particle offsets in subhalos
    g = h5py.File(basePath[:-6] + 'postprocessing/offsets/offsets_' + str(start_snap).zfill(3) + '.hdf5','r')
    
    if not g.__contains__('Subhalo'):
        g.close()
        raise ValueError(f'No Subhalos at snapshot {start_snap}!')
        
    dmInSubOffset = g['Subhalo/SnapByType'][:,1]
    g.close()
    
    numDMInSubs = il.groupcat.loadSubhalos(basePath, start_snap, fields = ['SubhaloLenType'])[:,1]
    shmr = il.groupcat.loadSubhalos(basePath, start_snap, fields = ['SubhaloHalfmassRadType'])[:,4] #ckpc/h
    
    cut = 2 #only trace back DM particles within two stellar halfmass radii
    
    which_particles, inside_offsets = dm.DM_in_2_shmr(DM_coords, sub_pos, dmInSubOffset, numDMInSubs, shmr, cut, boxSize) #do all calculations in ckpc/h!
    
    trace_back_indices = np.nonzero(which_particles)[0]
    
    if trace_back_indices.shape[0] == 0:
        raise Exception('No particles to trace back!')
        
    del DM_coords, sub_pos, shmr, which_particles
    
    ##### now trace those dm particles back: #####
    # only one 'tracer' per dm particle (IDs don't change with time)
    
    dmIDs = il.snapshot.loadSubset(basePath, start_snap, 1, fields = ['ParticleIDs'])
    dmIDs = dmIDs[trace_back_indices] #IDs to find at target snapshot
    
    dmIDs_target_snap = il.snapshot.loadSubset(basePath, target_snap, 1, fields = ['ParticleIDs'])
    
    _, target_DM_inds = dm.match_general(dmIDs, dmIDs_target_snap, is_sorted = False)
    
    f = h5py.File('/vera/ptmp/gc/olwitt/dm/' + basePath[32:39] + f'/dm_indices_{target_snap}.hdf5','w')
    f.create_dataset('dm_indices', data = target_DM_inds)
    f.create_dataset('dmInSubOffset', data = inside_offsets)
    f.close()
    done = time.time()
    print('time to run: ', done - start)
    return #target_DM_inds, inside_offsets

@jit(nopython=True, parallel=True)
def searchsorted_dm(dm_indices, dmInSubOffset_target_snap, numDMInSub_target_snap):
    
    location = np.empty(dm_indices.shape[0], np.intp)
    location.fill(-1)
    
    for i in nb.prange(dm_indices.shape[0]):
        ind = np.searchsorted(dmInSubOffset_target_snap,int(dm_indices[i]),'right')
        ind -= 1 #compensate for how np.searchsorted chooses the index

        #only important for last galaxy of halo: check whether particle is inner fuzz, i.e. IGM particles
        if dmInSubOffset_target_snap[ind] + numDMInSub_target_snap[ind] < dm_indices[i]:
            continue
                
        location[i]=ind
    return location

def location_of_dm(basePath, start_snap, target_snap, tree_ids,\
                        dm_indices, help_offsets, sub_ids, dmInSubOffset_target_snap,\
                        numDMInSub_target_snap, GFS_target_snap):
    """first output returns the subhalo index if the dm particle sits in a galaxy or -1 if it's in the IGM
    second output states, whether dm particle is inside main progenitor at target snapshot"""
    
    start = time.time()
        
    #find dm index in offset files in NlogM (for ALL subhalos)
    location = dm.searchsorted_dm(dm_indices, dmInSubOffset_target_snap, numDMInSub_target_snap)
    
    time_locating = time.time()
    
    #now identify parents that are still in their (main progenitor) galaxy or in one of its satellites
    
    isInMP = np.full(dm_indices.shape[0], 0, dtype = np.ubyte)
    isInMP_sat = np.full(dm_indices.shape[0], 0, dtype = np.ubyte)
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
        dmIndicesInSub = np.arange(help_offsets[j], help_offsets[j + 1]).astype(int)
        
        if j in missing or dmIndicesInSub.size == 0: #if subhalo hast no tree or no particles, skip it and assign "False"
#             if j in missing:
#                 test += dmIndicesInSub.size
            continue
        if tree_ids[sub_ids[j]]['SubfindID'].shape[0] <= start_snap - target_snap: #if tree doesn't reach until target_snap
            test += dmIndicesInSub.size            
            continue
            
        main_prog = tree_ids[sub_ids[j]]['SubfindID'][start_snap - target_snap]
        main_prog_central_index = np.where(GFS_target_snap == main_prog)[0]
        where_mp = np.where(location[dmIndicesInSub] == main_prog)[0] + dmIndicesInSub[0]
        isInMP[where_mp] = 1
        
        if main_prog_central_index.size > 0:
            if main_prog == max_central_id:
                next_central = numDMInSub_target_snap.shape[0]
            else:
                next_central = GFS_target_snap[main_prog_central_index + 1]
            where_mp_sat = np.where(np.logical_and(location[dmIndicesInSub] > main_prog,\
                                                   location[dmIndicesInSub] < next_central))[0] + dmIndicesInSub[0]
            isInMP_sat[where_mp_sat] = 1
        #print('main progenitor: ', main_prog)
        #print(location[parentIndicesInSub][:10],location[parentIndicesInSub][-10:])
        
        if target_snap == 99:
            assert (np.where(location[dmIndicesInSub] == main_prog)[0].shape[0]) == dmIndicesInSub.shape[0],\
            'offsets wrong probably'   
            assert np.all(isInMP[dmIndicesInSub] == 1), 'offsets wrong probably'
    
#     if target_snap == 99:
#         print(test)
    
    time_isInMP = time.time()
    return location, isInMP, isInMP_sat, time_locating - start, time_isInMP - time_locating

def save_location_dm(basePath, start_snap = 99):
    start = time.time()
    h_const = il.groupcat.loadHeader(basePath, start_snap)['HubbleParam']

    num_subs = il.groupcat.loadSubhalos(basePath, start_snap, fields = ['SubhaloMass']).shape[0]
    sub_ids = np.arange(num_subs)
    
    #load all MPBs
    tree_ids = loadMPBs(basePath, start_snap, ids = sub_ids, fields=['SubfindID'])

    stype = 'dm'
    
    file = '/vera/ptmp/gc/olwitt/' + stype + '/' + basePath[32:39] + f'/dm_indices_{start_snap}.hdf5'
    f = h5py.File(file,'r')
    dmInSubOffset = f['dmInSubOffset'][:]
    f.close()
    
    #check lowest saved parent index table snapshot
    min_snap = funcs.find_file_with_lowest_number('/vera/ptmp/gc/olwitt/' + stype + '/' + basePath[32:39])
    snaps = np.arange(99,min_snap - 1,-1)
    
    n = snaps.size    
    
    tot_time = 0
    tot_time_locating = 0
    tot_time_isInMP = 0
    
    save_file = h5py.File('/vera/ptmp/gc/olwitt/' + stype + '/' + basePath[32:39] + '/subhalo_index_table.hdf5','w')
    
    for i in range(n):
        start_loop = time.time()
        
        file = '/vera/ptmp/gc/olwitt/' + stype + '/' + basePath[32:39] + f'/dm_indices_{snaps[i]}.hdf5'
        f = h5py.File(file,'r')
        dm_indices = f['dm_indices'][:]       
        f.close()    
        
        g = h5py.File(basePath[:-6] + 'postprocessing/offsets/offsets_' + str(snaps[i]).zfill(3) + '.hdf5','r')
        if 'Subhalo' in list(g.keys()): # equivalent to g.__contains__('Subhalo')
            dmInSubOffset_target_snap = g['Subhalo/SnapByType'][:,1]
        else:
            g.close()
            continue
        g.close()
        
        numDMInSub_target_snap = il.groupcat.loadSubhalos(basePath, snaps[i],fields=['SubhaloLenType'])[:,1]
        GFS = il.groupcat.loadHalos(basePath, snaps[i], fields = ['GroupFirstSub'])
        
        grp = save_file.create_group('snap_'+ str(snaps[i]))
        
        #run function to determine location of parents
        location, isInMP, isInMP_sat, time_locating, time_isInMP = \
        dm.location_of_dm(basePath, start_snap = 99, target_snap = snaps[i], tree_ids = tree_ids,\
                            dm_indices = dm_indices, help_offsets = dmInSubOffset, sub_ids = sub_ids,\
                            dmInSubOffset_target_snap = dmInSubOffset_target_snap,\
                           numDMInSub_target_snap = numDMInSub_target_snap, GFS_target_snap = GFS)
        
        grp.create_dataset('location', data = location)
#         grp.create_dataset('isInMP', data = isInMP)
#         grp.create_dataset('isInMP_satellite', data = isInMP_sat)
            
        isInCentral = np.full(location.shape[0], 0, dtype = int)

#         _,_,central_indices = np.intersect1d(GFS[np.where(GFS != -1)],location, return_indices = True)
        central_indices = np.nonzero(np.isin(location, GFS[np.where(GFS != -1)]))[0]
        isInCentral[central_indices] = 1 #mark which parents are located within a central galaxy with 1
        
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
        
        del isInCentral, isInMP, isInMP_sat, location, GFS, dm_indices
        
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

def dm_accretion_channels(basePath, stype, sub_ids, start_snap = 99):
    start = time.time()
    h_const = il.groupcat.loadHeader(basePath, start_snap)['HubbleParam']
    
    #check lowest saved parent index table snapshot
    min_snap = funcs.find_file_with_lowest_number('/vera/ptmp/gc/olwitt/' + stype + '/' + basePath[32:39])
    snaps = np.arange(99,min_snap - 1,-1)
    
    n = snaps.size
    
    z = np.zeros(n)

    file = '/vera/ptmp/gc/olwitt/' + stype + '/' + basePath[32:39] + f'/dm_indices_{start_snap}.hdf5'
    f = h5py.File(file,'r')   

    dm_indices = f['dm_indices'][:]
    dmInSubOffset = f['dmInSubOffset'][:]
    num_tracers = dm_indices.shape[0]
    del dm_indices
    
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
    max_ind = max(dmInSubOffset)
    for i in range(sub_ids[-1]):
        index = dmInSubOffset[i]
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
        indcs = np.arange(dmInSubOffset[sub_ids[i-1]],dmInSubOffset[sub_ids[i-1]+1])
        if indcs.size > 0 and sub_ids[i-1] not in missing:
            isGalaxy[i-1] = True
        which_indices[counter:counter + indcs.shape[0]] = indcs
        help_offsets[i-1] = indcs.shape[0]
        counter += indcs.shape[0]
        
    del indcs, counter, isInMP
    
    #trim zeros at the end:
    which_indices = np.trim_zeros(which_indices,'b').astype(int)
    
    #compute correct offsets:
    ## states, which indices correspond to which subhalo from sub_ids
    help_offsets = np.cumsum(help_offsets).astype(int)
    help_offsets = np.insert(help_offsets,0,0)
    
    location_file = h5py.File('/vera/ptmp/gc/olwitt/' + stype + '/' + basePath[32:39] + '/subhalo_index_table.hdf5','r')
    
    #initialize output arrays for the entire simulation -> #entries = #snapshots
    mp = np.zeros(n)
    igm = np.zeros(n)
    sub = np.zeros(n)
    other_centrals = np.zeros(n)
    other_satellites = np.zeros(n)
    mp_satellites = np.zeros(n)
    
    total = np.zeros(n)

    nums = np.zeros((n,5,8))
    gal_comp = np.zeros((n,sub_ids.shape[0],8)) #galaxy composition
    
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
        file = '/vera/ptmp/gc/olwitt/' + stype + '/' + basePath[32:39] + f'/dm_indices_{snaps[i]}.hdf5'
        f = h5py.File(file,'r')      

        dm_indices = f['dm_indices'][:]
        f.close()
        if i==1:
            start_loop = time.time()
        
        #only consider indices of relevant galaxies        
        dm_indices = dm_indices[which_indices]
        
        if i==1:
            end_files = time.time()
            print('time for loading snap from files: ',end_files-start_loop, flush = True)
        
        before_locating = time.time()
        
        if not location_file.__contains__(f'snap_{snaps[i]}'): #skip snapshot if there are no datasets in hdf5 file
            continue
        
        #load location of parents from file (but only particles from subhalos that we are interested in)
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
        
        if i==0:
            print(f'{np.where(isInMP == 0)[0].shape[0]} tracers not in MP at z=0')
            print(f'{np.where(location == -1)[0].shape[0]} tracers in the IGM at z=0')
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
        nums[i,:,:], gal_comp[i,:,:] = dm.binning(dm_indices, location, isInMP, isInMP_sat, isInCentral, sub_ids, help_offsets,\
                                                  mass_bin1, mass_bin2, mass_bin3, mass_bin4, mass_bin5)
        
        mp[i] = np.where(isInMP == 1)[0].shape[0] #number of particles in the MP 
        igm[i] = np.where(location == -1)[0].shape[0] #number of particles in the IGM
        
        mp_satellites[i] = np.where(isInMP_sat == 1)[0].shape[0] #number of particles in satellites of the MP
        
        #number of particles in satellites of other central galaxies than the MP
        other_satellites[i] = np.where(np.logical_and(np.logical_and(location != -1, isInCentral == 0),\
                                                      isInMP == 0))[0].shape[0] - mp_satellites[i]
        
        #number of particles in other central galaxies than the MP (other halos)
        other_centrals[i] = np.where(np.logical_and(np.logical_and(location != -1, isInCentral == 1),isInMP == 0))[0].shape[0]
        
        #number of particles in other galaxies than the MP; satellites + other_centrals = sub
        sub[i] = np.where(location != -1)[0].shape[0] - mp[i] #everything not in the igm is in a subhalo (or a FoF halo)
        
        z[i] = il.groupcat.loadHeader(basePath, snaps[i])['Redshift'] 
        total[i] = igm[i] + sub[i] + mp[i]
        
        ##### what fraction of baryonic matter forming in situ stars by z=0 was accreted in a certain way? #####
        if i > 0: #skip start_snapshot
            # mark each particle currently in another galaxy than the MP:
            other_gal_tmp = np.where(np.logical_and(location != -1, isInMP == 0))[0] 
            directly_from_igm[other_gal_tmp] = 0 #assume all came from igm, delete those that were in other galaxies
            
            # smooth accretion (was in IGM, now in MP)
            smooth_tmp = np.where(np.logical_and(np.logical_and(old_isInMP == 1, location == -1), isInMP == 0))[0] 
            smooth[smooth_tmp] = 1 #assume none were smoothly accreted, mark those that are
            
            # mark each particle currently in another halo than the one of the MP (= intergalactic transfer)
            other_halos_tmp = np.where(np.logical_and(np.logical_and(location != -1, isInMP == 0), isInMP_sat == 0))[0] 
            from_other_halos[other_gal_tmp] = 1 #assume none from other halos (or galaxies in general, respectively), mark those that are 
            
            # wind recycling (was in MP, now in IGM, and eventually in MP again (z=0))
            wind_rec_tmp = np.where(np.logical_and(old_location == -1, isInMP == 1))[0] 
            long_range_wind_recycled[wind_rec_tmp] = 1
            
            # mark each particle, that was bound to another subhalo prior to being in the MP
            merger_tmp = np.where(np.logical_and(np.logical_and(isInMP == 0, location != -1), old_isInMP == 1))[0]
            mergers[merger_tmp] = 1
            
            # mark particles, that entered the MP via a merger BEFORE they were accreted smoothly (e.g. due to wind recycling)
            merger_first_tmp = np.where(np.logical_and(smooth == 0, mergers == 1))[0]
            merger_before_smooth[merger_first_tmp] = 1
            
        
        old_location = location.copy() #old_location refers to higher snapshot (later time) than location, as snaps[i] decreases in each loop
        old_isInMP = isInMP.copy()
#         old_isInMP_sat = isInMP_sat.copy()
        
        if(i==1):
            end_binning = time.time()
            print('total time for binning: ', end_binning-end_create_bins, flush = True)
            print('total time for first loop: ', end_binning-start_loop, flush = True)
        print(snaps[i], 'done;', end = ' ', flush = True)
    print()
    
    #find all particles that were in halos at some point AND meet the smooth accretion criterion
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
    
    assert np.nonzero(smooth)[0].shape[0] == np.nonzero(directly_from_igm)[0].shape[0] - alwaysInMP.shape[0] +\
    np.nonzero(stripped_from_halos)[0].shape[0] + np.nonzero(merger_before_wind_rec)[0].shape[0]
    
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
    
    # only use galaxies with at least one particle and existing tree
    no_gal = np.nonzero(np.logical_not(isGalaxy))[0]
    gal_comp[:,no_gal,:] = -1
    gal_accretion_channels[no_gal,:] = -1
    location_file.close()
    return mp, igm, sub, other_satellites, mp_satellites, other_centrals, total, nums, z, gal_comp, isGalaxy, directly_from_igm, stripped_from_halos, from_other_halos, mergers, long_range_wind_recycled, nep_wind_recycled, gal_accretion_channels

@njit
def binning(dm_indices, location, isInMP, isInMP_sat, isInCentral, sub_ids, help_offsets, mass_bin1, mass_bin2, mass_bin3, mass_bin4, mass_bin5):
    res = np.zeros((5,8))
    gal_res = np.zeros((help_offsets.shape[0] - 1,8))
    
    #determine mass fractions for every galaxy individually
    for i in range(0,help_offsets.shape[0] - 1):
        indices = np.arange(help_offsets[i],help_offsets[i+1])
        
        gal_res[i,0] = np.where(isInMP[indices] == 1)[0].shape[0] #number of particles in the MP
        gal_res[i,1] = -1 #not in use
        gal_res[i,2] = -1 #not in use
        
        gal_res[i,3] = np.where(location[indices] != -1)[0].shape[0] - gal_res[i,0] # number of particles in other galaxies 
        gal_res[i,5] = np.where(isInMP_sat[indices] == 1)[0].shape[0] #number of particles in satellites of the MP
        gal_res[i,4] = np.where(np.logical_and(np.logical_and(location[indices] != -1, isInCentral[indices] == 0), isInMP[indices] == 0))[0].shape[0] - gal_res[i,5] #number of particles in satellites of other halos
        
        gal_res[i,6] = np.where(np.logical_and(np.logical_and(location[indices] != -1, isInCentral[indices] == 1),isInMP[indices] == 0))[0].shape[0] #number of particles in other centrals (halos)
        
        gal_res[i,7] = indices.shape[0] #total
        gal_res[i,:7] = gal_res[i,:7] / gal_res[i,7] if gal_res[i,7] > 0 else gal_res[i,:7] #obtain mass fractions
    
    #determine mass fractions for entire mass bins
    for i in nb.prange(5):
        mass_bin = mass_bin1 if i==0 else mass_bin2 if i==1 else mass_bin3 if i==2 else mass_bin4 if i==3 else\
        mass_bin5
        indices = np.nonzero(funcs.isin(location,mass_bin))[0]
        
        res[i,0] = np.where(isInMP[indices] == 1)[0].shape[0] #number of particles in the MP
        res[i,1] = -1 #not in use
        res[i,2] = -1 #not in use
        
        res[i,3] = np.where(location[indices] != -1)[0].shape[0] - res[i,0] # number of particles in other galaxies 
        res[i,5] = np.where(isInMP_sat[indices] == 1)[0].shape[0] #number of particles in satellites of the MP
        res[i,4] = np.where(np.logical_and(np.logical_and(location[indices] != -1, isInCentral[indices] == 0), isInMP[indices] == 0))[0].shape[0] - res[i,5] #number of particles in satellites of other halos
        
        res[i,6] = np.where(np.logical_and(np.logical_and(location[indices] != -1, isInCentral[indices] == 1),isInMP[indices] == 0))[0].shape[0] #number of particles in other centrals (halos)
        
        res[i,7] = indices.shape[0] #total
        res[i,:7] = res[i,:7] / res[i,7] if res[i,7] > 0 else res[i,:7] #obtain mass fractions
    return res, gal_res

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

@jit(nopython = True, parallel = True)
def num_DM_in_2_shmr(DM_coords, sub_pos, dmInSubOffset, shmr, cut, boxSize):
    """ Outputs the fraction of DM particles inside cut*shmr of all subhalos at a specific snapshot."""
    frac_in_2_shmr = np.zeros(sub_pos.shape[0])
    
    for i in nb.prange(sub_pos.shape[0]):
        sub_indices = np.arange(dmInSubOffset[i], dmInSubOffset[i+1]) #particle indices in subhalo
        sub_distances_dm = funcs.dist_vector_nb(sub_pos[i], DM_coords[sub_indices], boxSize) #particle distances to sub center
        inside = np.where(sub_distances_dm <= cut * shmr[i])[0] #particles inside 2shmr
        frac_in_2_shmr[i] = inside.shape[0] / sub_indices.shape[0] if sub_indices.shape[0] > 0 else 0
    
    return frac_in_2_shmr

def dm_halo_core_fractions(basePath, start_snap, target_snap):
    
    start = time.time()
    h = il.groupcat.loadHeader(basePath, start_snap)
    h_const = h['HubbleParam']
    boxSize = h['BoxSize']
    num_subs = h['Nsubgroups_Total']

#     num_subs = il.groupcat.loadSubhalos(basePath, start_snap, fields = ['SubhaloMass']).shape[0]
    sub_ids = np.arange(num_subs)
    central_sub_ids = il.groupcat.loadHalos(basePath, start_snap, fields = ['GroupFirstSub'])
    central_sub_ids = central_sub_ids[np.where(central_sub_ids != -1)]
    
    stype = 'dm'
    
    #check lowest saved dm index table snapshot
#     min_snap = funcs.find_file_with_lowest_number('/vera/ptmp/gc/olwitt/dm/' + basePath[32:39])
    min_snap = 0
    snaps = np.arange(99,min_snap - 1,-1)
    
    n = snaps.size
    
    #mark physical subhalos with good properties at target snapshot
    subhaloFlag = np.ones(sub_ids.shape[0], dtype = np.ubyte)
    central_subhaloFlag = np.ones(sub_ids.shape[0], dtype = np.ubyte)
    
    #mark satellites as not suitable
    central_subhaloFlag[np.isin(sub_ids,central_sub_ids, invert = True)] = 0
    
    #load all MPBs of SHMR and the SubfindID
    trees = loadMPBs(basePath, start_snap, ids = sub_ids, fields = ['SubhaloHalfmassRadType', 'SubfindID'])
        
    tree_check = list(trees)
    
    #determine missing trees:
    missing = []
    counter = 0
    
    for i in range(sub_ids[-1]):
        if i != tree_check[counter]:
            missing.append(i)
            subhaloFlag[i] = 0
            central_subhaloFlag[i] = 0
            i += 1
            continue
        counter += 1
#     subhaloFlag[missing] = -1    
    
    sub_positions = h5py.File('files/'+basePath[32:39]+'/SubhaloPos_new_extrapolated.hdf5','r')
    is_extrapolated = sub_positions['is_extrapolated'][:] #only use subhalos with extrapolated position history
    subhaloFlag[np.where(is_extrapolated == False)] = 0
    del is_extrapolated

#     sub_pos_at_target_snap = il.groupcat.loadSubhalos(basePath, target_snap, fields = ['SubhaloPos'])

    sub_pos_at_target_snap = sub_positions['SubhaloPos'][:,start_snap - target_snap,:] #ckpc/h
    sub_positions.close()
    
    pre_loop = time.time()
    print('pre-loop loading done in ', pre_loop - start)
        
    loop_start = time.time()

    #load stellar halfmass radii -> if tree has not sufficient number of entries: put 0 as SHMR -> no particle within galaxy
    shmr_at_target_snap = np.zeros(num_subs)
    target_sub_ids = np.full(num_subs, -2, dtype = int)
    for j in range(num_subs):
        #if subhalo tree reaches until target_snap
        if j not in missing and start_snap - target_snap < trees[sub_ids[j]]['count']: 
            shmr_at_target_snap[j] = trees[sub_ids[j]]['SubhaloHalfmassRadType'][start_snap - target_snap][4] #ckpc/h
            target_sub_ids[j] = trees[sub_ids[j]]['SubfindID'][start_snap - target_snap]
        else:
            shmr_at_target_snap[j] = 0 #ckpc/h
            
            ##################################################################
            if target_snap > 12: #mark tree as difficult to use only for z <= 6
                subhaloFlag[j] = 0
            ##################################################################
    
    #exclude all galaxies which aren't centrals anymore from the R_vir criterion computation
    groupFirstSub = il.groupcat.loadHalos(basePath, target_snap, fields = ['GroupFirstSub'])
    r_vir = np.full(sub_ids.shape[0], 0)
    if type(groupFirstSub) == dict:
        central_subhaloFlag[:] = 0              
    else:
        central_sub_ids_at_target_snap, GFS_inds, TSID_inds = np.intersect1d(groupFirstSub, target_sub_ids, return_indices = True)
        r_vir_cat = il.groupcat.loadHalos(basePath, target_snap, fields = ['Group_R_Crit200'])[GFS_inds] #ckpc/h
        r_vir = np.full(sub_ids.shape[0], 0)
        r_vir[TSID_inds] = r_vir_cat 
        
        # all subhalos that are no central anymore are assigned r_vir=0...
        mask = np.full(num_subs, True, dtype = bool)
        mask[TSID_inds] = False
        
        #... and are marked as not suitable for analysis
        central_subhaloFlag[mask] = 0
        del mask, TSID_inds, r_vir_cat, groupFirstSub, central_sub_ids_at_target_snap, GFS_inds
    
    shmr_load = time.time()
    print('time for SHMR loading: ', shmr_load - loop_start)
    #load coordinates of dm particles (from snapshot files) and subhalos (from extrapolation files)

    file = '/vera/ptmp/gc/olwitt/' + stype + '/' + basePath[32:39] + f'/dm_indices_{target_snap}.hdf5'
    f = h5py.File(file,'r')  
    dm_indices = f['dm_indices'][:]
    dmInSubOffset = f['dmInSubOffset'][:]
    f.close()    
    
    counter = 0
    for i in range(1,sub_ids.shape[0] + 1):
        indcs = np.arange(dmInSubOffset[sub_ids[i-1]],dmInSubOffset[sub_ids[i-1]+1])
        if indcs.size > 0 and sub_ids[i-1] not in missing:
            counter += 1
        else:
            subhaloFlag[i-1] = 0
            
    print('# good galaxies: ', counter)

    
#         dm_coords = sm.loadSubset(basePath, target_snap, 1, subset = dm_indices, is_sorted = False, fields = ['Coordinates']) #ckpc/h
    dm_coords = il.snapshot.loadSubset(basePath, target_snap, 1, fields = ['Coordinates'])[dm_indices,:] #ckpc/h

    cut = 2 #number of R_0.5,star defining the region of interest around the subhalo center

    coord_load = time.time()
    print('time for coordinate loading: ', coord_load - shmr_load)

    #this gives the fraction of dm particles within the galaxy (2*R_0.5,star) for every subhalo at a specific snapshot
    frac_in_2_shmr = dm.num_DM_in_2_shmr(dm_coords, sub_pos_at_target_snap, dmInSubOffset, shmr_at_target_snap, cut, boxSize)
    frac_in_r200 = dm.num_DM_in_2_shmr(dm_coords, sub_pos_at_target_snap, dmInSubOffset, r_vir, 1, boxSize)

    loop_end = time.time()
    print('loop finished in ', loop_end - loop_start)

    #save core fraction array:
    
    f = h5py.File('/vera/ptmp/gc/olwitt/dm/' + basePath[32:39] + f'/core_fractions/core_fractions_{target_snap}.hdf5','w')
    f.create_dataset(f'core_fractions_2SHMR_{target_snap}', data = frac_in_2_shmr)
    f.create_dataset(f'subhaloFlag_{target_snap}', data = subhaloFlag)
    f.create_dataset(f'core_fractions_R200_{target_snap}', data = frac_in_r200)
    f.create_dataset(f'central_subhaloFlag_{target_snap}', data = central_subhaloFlag)
    f.close()
    
    print(f'snap {target_snap} done;', end = ' ')
    return 


def dm_halo_core_formation_time(basePath, start_snap):
    
    start = time.time()
    h = il.groupcat.loadHeader(basePath, start_snap)
    h_const = h['HubbleParam']
    boxSize = h['BoxSize']
    num_subs = h['Nsubgroups_Total']
    z = iF.give_z_array(basePath)
    
    core_form_snap = np.zeros(num_subs, dtype = np.ubyte) #result array
    form_snap = np.zeros(num_subs, dtype = np.ubyte)
    subhaloFlag = np.ones(num_subs, dtype = np.ubyte)
    central_subhaloFlag = np.ones(num_subs, dtype = np.ubyte)
    
    #check lowest saved dm index table snapshot
#     min_snap = funcs.find_file_with_lowest_number('/vera/ptmp/gc/olwitt/dm/' + basePath[32:39] + '/core_fractions')
    min_snap = 0
    snaps = np.arange(99,min_snap - 1,-1)
    
    for i, target_snap in enumerate(snaps):
    
        start_loop = time.time()
        f = h5py.File('/vera/ptmp/gc/olwitt/dm/' + basePath[32:39] + f'/core_fractions/core_fractions_{target_snap}.hdf5','r')
        frac_in_2_shmr = f[f'core_fractions_2SHMR_{target_snap}'][:]
        tmp_subhaloFlag = f[f'subhaloFlag_{target_snap}'][:]
        frac_in_r_vir = f[f'core_fractions_R200_{target_snap}'][:]
        tmp_central_subhaloFlag = f[f'central_subhaloFlag_{target_snap}'][:]
        f.close()
        
        if i > 12:
            subhaloFlag[np.where(tmp_subhaloFlag == 0)] = 0
            central_subhaloFlag[np.where(tmp_central_subhaloFlag)] = 0
    
        if i > 0:
            # there might be the possibility, that such a change in the fraction of dark matter particles within 2SHMR occurs more than once
            # for a single galaxy. In this case, the algorithm captures the earliest time (lowest snapshot) at which this happens. This is
            # also what we want, as we're looking for the first time, when this threshold is reached. This point in time marks the 
            # formation snapshot of the subhalo.
            core_formation_indices = np.where(np.logical_and(old_frac_in_2_shmr > 0.5, frac_in_2_shmr <= 0.5))[0]
            formation_indices = np.where(np.logical_and(old_frac_in_r_vir > 0.5, frac_in_r_vir <= 0.5))
            core_form_snap[core_formation_indices] = target_snap
            form_snap[formation_indices] = target_snap
        
        old_frac_in_2_shmr = frac_in_2_shmr.copy() #old means in the previous loop, i.e. at a higher snapshot
        old_frac_in_r_vir = frac_in_r_vir.copy()
        end_loop = time.time()
        if i== 1:
            print('time for loop: ', end_loop - start_loop)
        
    core_form_z = z[99 - core_form_snap]
    form_z = z[99 - form_snap]
    
    f = h5py.File('files/' + basePath[32:39] + f'/dm_halo_core_formation_times.hdf5','w')
    f.create_dataset('core_formation_times', data = core_form_z)
    f.create_dataset('formation_times', data = form_z)
    f.create_dataset('subhaloFlag', data = subhaloFlag)
    f.create_dataset('central_subhaloFlag', data = central_subhaloFlag)
    f.close()
    
    return

#new function to use the output from lagrangian_region_times.py
def dm_halo_core_formation_time_lagr_reg(basePath, start_snap):
    
    start = time.time()
    h = il.groupcat.loadHeader(basePath, start_snap)
    num_subs = h['Nsubgroups_Total']
    print(num_subs)
    z = iF.give_z_array(basePath)
    run = basePath[38]
    
    shmr_thresh = 2
    r_vir_thresh = 1
    
    core_form_snap = np.full(num_subs, -1, dtype = np.byte) #result array
    form_snap = np.full(num_subs, -1, dtype = np.byte)
    subhaloFlag = np.zeros(num_subs, dtype = np.ubyte)
    core_subhaloFlag = np.zeros(num_subs, dtype = np.ubyte)
    last_central_snap = np.full(num_subs, -1, dtype = np.ubyte)
    
    min_snap = 0
    snaps = np.arange(99,min_snap - 1,-1)
    
    #determine, at which snapshot each subhalo is no longer a central
    for target_snap in snaps:
    
        file = f'/vera/ptmp/gc/olwitt/dm/TNG50-{run}/lagrangian_regions/lagrangian_regions_cut21_{target_snap}.hdf5'
        f = h5py.File(file,'r')
        tmp_subhaloFlag = f['subhaloFlag'][:]
#         print(tmp_subhaloFlag.shape)
        f.close()
        
        #only consider subhalos as centrals at a certain snapshot, if they were a central in the previously considered snapshot
        if target_snap == start_snap:
            last_central_snap[np.where(tmp_subhaloFlag == 1)] = target_snap
        else:
            last_central_snap[np.where(np.logical_and(tmp_subhaloFlag == 1, old_tmp_subhaloFlag == 1))] = target_snap
        
        old_tmp_subhaloFlag = tmp_subhaloFlag.copy()
    
    
    for i, target_snap in enumerate(snaps):
    
        start_loop = time.time()
        file = f'/vera/ptmp/gc/olwitt/dm/TNG50-{run}/lagrangian_regions/lagrangian_regions_cut21_{target_snap}.hdf5'
        f = h5py.File(file,'r')
        
        #subhalo medians are NaN if not computed (bc not a central or something)
#         print(f['lagrangian_regions_shmr'].shape)
        if len(f['lagrangian_regions_shmr'].shape) == 2:
            sub_med_dist_shmr = f['lagrangian_regions_shmr'][:,0]
            sub_med_dist_r_vir = f['lagrangian_regions_r_vir'][:,0]
            f.close()
            print(i)
        else:
            f.close()
            continue
        
    
        if i > 0:
            # there might be the possibility, that such a change in the fraction of dark matter particles within 2SHMR occurs more than once
            # for a single galaxy. In this case, the algorithm captures the earliest time (lowest snapshot) at which this happens. This is
            # also what we want, as we're looking for the first time, when this threshold is reached. This point in time marks the 
            # formation snapshot of the subhalo.
            #problem: not calculated: NaN. But: NaN <,==,> smth is always False
            with np.errstate(invalid='ignore'):
                core_formation_indices = np.where(np.logical_and(old_sub_med_dist_shmr <= shmr_thresh, sub_med_dist_shmr >= shmr_thresh))[0]
                formation_indices = np.where(np.logical_and(old_sub_med_dist_r_vir <= r_vir_thresh, sub_med_dist_r_vir >= r_vir_thresh))[0]
            
#             print(formation_indices.shape, core_formation_indices.shape)
            
            core_form_snap[core_formation_indices] = target_snap + 1
            form_snap[formation_indices] = target_snap + 1
            
            # only mark those subhalos as suitable for analysis that were a central before they formed there dm core
            #possible that this has to be repeated for formation_indices
            core_subhaloFlag[np.where(np.logical_and(last_central_snap[core_formation_indices] <= target_snap + 1,\
                                               last_central_snap[core_formation_indices] != -1))] = 1
            subhaloFlag[np.where(np.logical_and(last_central_snap[formation_indices] <= target_snap + 1,\
                                               last_central_snap[formation_indices] != -1))] = 1
        
        old_sub_med_dist_shmr = sub_med_dist_shmr.copy() #old means in the previous loop, i.e. at a higher snapshot
        old_sub_med_dist_r_vir = sub_med_dist_r_vir.copy()
        
        
    core_form_z = np.full(num_subs, -1, dtype = np.float)
    mask = np.where(core_form_snap != -1)
    core_form_z[mask] = z[99 - core_form_snap[mask]]
    form_z = np.full(num_subs, -1, dtype = np.float)
    mask = np.where(form_snap != -1)
    form_z[np.where(form_snap[mask] != -1)] = z[99 - form_snap[mask]]
    
    end = time.time()
    print('total time: ', end - start)
    
    file = '/vera/ptmp/gc/olwitt/auxCats/' + basePath[32:39] + f'/dm_halo_core_formation_times.hdf5'
    f = h5py.File(file,'w')
    f.create_dataset('core_formation_times', data = core_form_z)
    f.create_dataset('formation_times', data = form_z)
    f.create_dataset('core_subhaloFlag', data = core_subhaloFlag)
    f.create_dataset('subhaloFlag', data = subhaloFlag)
    f.close()
    
    return

@jit(nopython = True, parallel = True)
def isSubGalaxy(sub_ids, extrapolated_sub_ids, final_offsets):
    isGalaxy = np.full(extrapolated_sub_ids.shape[0], False)
    for i in nb.prange(extrapolated_sub_ids.shape[0]):
        sub_id = sub_ids[extrapolated_sub_ids[i]]
        if final_offsets[sub_id + 1] - final_offsets[sub_id] < 2:
            continue
        isGalaxy[i] = True
    return isGalaxy

@jit(nopython = True, parallel = True)
def distances(dm_indices, location_at_cut, isInMP_at_cut, final_offsets, all_dm_pos, sub_pos_at_target_snap, extrapolated_sub_ids,\
     extrapolated_central_sub_ids, sub_ids, r_norm , r_vir, boxSize):
    
    sub_medians = np.full((extrapolated_sub_ids.shape[0],3),np.nan)
    sub_medians_r_vir = np.full((extrapolated_sub_ids.shape[0],3),np.nan)
      
    for index in nb.prange(extrapolated_sub_ids.shape[0]):
        if extrapolated_sub_ids[index] == -1: #stop computation if r_norm = 0 or no subhalo position history
            continue
        sub_id = sub_ids[extrapolated_sub_ids[index]]
        indices_of_sub = np.arange(final_offsets[sub_id],final_offsets[sub_id+1])
        location_of_sub_at_cut = location_at_cut[indices_of_sub]
        isInMP_of_sub_at_cut = isInMP_at_cut[indices_of_sub]
        
        dm_indices_of_sub = dm_indices[indices_of_sub]

        particle_pos = all_dm_pos[dm_indices_of_sub,:]
        
        subhalo_position = sub_pos_at_target_snap[sub_id,:] #prior: sub_id instead of index!!!

        rad_dist = funcs.dist_vector_nb(subhalo_position,particle_pos,boxSize)
        
#         if index in [5,10,17]:
#             print(sub_id)
#             print(rad_dist[:3]/ r_norm[index], np.min(rad_dist)/r_norm[index], np.median(rad_dist) / r_norm[index], np.max(rad_dist)/r_norm[index])
        
        igm_mask = np.where(location_of_sub_at_cut == -1)[0]
        satellite_mask = np.where((location_of_sub_at_cut != -1) & (np.logical_not(isInMP_of_sub_at_cut)))[0]
                
        if rad_dist.size > 0:
            sub_medians[index,0] = np.median(rad_dist) / r_norm[index]
            
        if igm_mask.size > 0:
            sub_medians[index,1] = np.median(rad_dist[igm_mask]) / r_norm[index]

        if satellite_mask.size > 0:
            sub_medians[index,2] = np.median(rad_dist[satellite_mask]) / r_norm[index]
            
        
        if extrapolated_central_sub_ids[index] == -1: #stop computation if r_vir = 0 or galaxy isn't a central in this snapshot
            continue
        
        
        if rad_dist.size > 0:
            sub_medians_r_vir[index,0] = np.median(rad_dist) / r_vir[index]
            
        if igm_mask.size > 0:
            sub_medians_r_vir[index,1] = np.median(rad_dist[igm_mask]) / r_vir[index]

        if satellite_mask.size > 0:
            sub_medians_r_vir[index,2] = np.median(rad_dist[satellite_mask]) / r_vir[index]
            
        
    return sub_medians, sub_medians_r_vir

def lagrangian_region(basePath, stype, start_snap, target_snap, cut_snap, sub_ids, boxSize, r_norm_trees):
    start_loading = time.time()
    header = il.groupcat.loadHeader(basePath,target_snap)
    redshift = header['Redshift']
    h_const = header['HubbleParam']
    boxSize = header['BoxSize'] / h_const
    
    #load data from files ---------------------------------------------------------------------------------
    file = '/vera/ptmp/gc/olwitt/' + stype + '/' + basePath[32:39] + f'/dm_indices_{target_snap}.hdf5'
    f = h5py.File(file,'r')
    dm_indices = f['dm_indices'][:]
    dmInSubOffset = f['dmInSubOffset'][:]
    
    sub_positions = h5py.File('files/'+basePath[32:39]+'/SubhaloPos_new_extrapolated.hdf5','r') 
    is_extrapolated = sub_positions['is_extrapolated'][:]
    
    sub_pos_at_target_snap = sub_positions['SubhaloPos'][:,start_snap-target_snap,:] / h_const
    sub_positions.close()
    
#     parent_indices_data = parent_indices[:,:].astype(int)
    
    loc_file = h5py.File(f'/vera/ptmp/gc/olwitt/' + stype + '/' + basePath[32:39] + '/subhalo_index_table.hdf5','r')
    location = loc_file[f'snap_{cut_snap}/location'][:]
    
    location_type = loc_file[f'snap_{cut_snap}/location_type'][:]
    isInMP = np.zeros(location_type.shape[0], dtype = np.ubyte)
    isInMP[np.isin(location_type,np.array([1,2]))] = 1
    del location_type
    loc_file.close()
        
    all_dm_pos = il.snapshot.loadSubset(basePath,target_snap,'dm',fields=['Coordinates']) / h_const
    
    #which galaxies? ----------------------------------------------------------------------------------------
    extrapolated_sub_ids = np.where(is_extrapolated[sub_ids])[0]
    del is_extrapolated
    #test, which galaxies have zero tracers of insitu stars -> those won't have a meaningful radial profile
    # (this already excludes all galaxies without any stars, since they can't have insitu stars)    
    isGalaxy = isSubGalaxy(sub_ids, extrapolated_sub_ids, dmInSubOffset)
    
    #only use galaxies that have at least one tracer particle (at z=0) AND have an extrapolated SubhaloPos entry
    extrapolated_sub_ids = extrapolated_sub_ids[np.where(isGalaxy)[0]] #all galaxies without extrapolated sub_pos history or only 1 tracer: -1
    del isGalaxy
    
    #<until here, extrapolated_sub_ids is identical for every snapshot>
    
    #now aquire the correct virial radii (consider only those galaxies that are still centrals):
    r_norm = np.full(extrapolated_sub_ids.shape[0], -2) #-2 bc. GroupFirstSub could contain -1's
#     r_vir_tree = r_norm.copy()
    target_sub_ids = r_norm.copy()
#     sub_pos_at_target_snap = np.zeros((extrapolated_sub_ids.shape[0],3))
    for i in range(r_norm.shape[0]):
        if start_snap - target_snap < r_norm_trees[sub_ids[extrapolated_sub_ids[i]]]['count']: #if tree has sufficient entries
            r_norm[i] = r_norm_trees[sub_ids[extrapolated_sub_ids[i]]]['SubhaloHalfmassRadType'][start_snap - target_snap][4] / h_const
#             r_vir_tree[i] = r_norm_trees[sub_ids[extrapolated_sub_ids[i]]]['Group_R_Crit200'][start_snap - target_snap] / h_const
            target_sub_ids[i] = r_norm_trees[sub_ids[extrapolated_sub_ids[i]]]['SubfindID'][start_snap - target_snap]
#             sub_pos_at_target_snap[i] = r_norm_trees[sub_ids[extrapolated_sub_ids[i]]]['SubhaloPos'][start_snap - target_snap][:]/h_const
    del r_norm_trees
    
    #mark all galaxies which aren't centrals anymore
    groupFirstSub = il.groupcat.loadHalos(basePath, target_snap, fields = ['GroupFirstSub'])
    central_sub_ids_at_target_snap, GFS_inds, TSID_inds = np.intersect1d(groupFirstSub, target_sub_ids, return_indices = True)
    r_vir_cat = il.groupcat.loadHalos(basePath, target_snap, fields = ['Group_R_Crit200'])[GFS_inds]
    r_vir = np.full(extrapolated_sub_ids.shape[0], -2)
    r_vir[TSID_inds] = r_vir_cat
#     assert funcs.areEqual(r_vir, r_vir_tree), 'virial radii from trees differ to those from the groupcat!'
    
    tmp_extrapolated_sub_ids = extrapolated_sub_ids.copy()
    extrapolated_central_sub_ids = np.full(tmp_extrapolated_sub_ids.shape[0],-1)
    extrapolated_central_sub_ids[TSID_inds] = tmp_extrapolated_sub_ids[TSID_inds] #all galaxies without extrapolated sub_pos history or 
                                                                                    #only 1 tracer or being a satellite (at target_snap): -1
    del tmp_extrapolated_sub_ids, TSID_inds, GFS_inds, central_sub_ids_at_target_snap
    
    zero_r_norm = np.where(r_norm <= 0.0001)[0]
    zero_r_vir = np.where(r_vir <= 0.1)[0]
    #exclude all galaxies with normalization radius smaller than 0.0001ckpc, i.e. essentially zero ckpc
    extrapolated_sub_ids[zero_r_norm] = -1 #all galaxies without extrapolated sub_pos history or only 1 tracer or having a vanishing r_norm:
                                                                            # -1
    extrapolated_central_sub_ids[zero_r_vir] = -1 #all galaxies without extrapolated sub_pos history or only 1 tracer or having a 
                                                    #vanishing r_vir or being a satellite: -1
    
    
    start = time.time()
    print('time for loading and shit: ',start-start_loading)
    sub_medians, sub_medians_r_vir =\
    distances(dm_indices, location, isInMP, dmInSubOffset, all_dm_pos, sub_pos_at_target_snap, extrapolated_sub_ids,\
              extrapolated_central_sub_ids, sub_ids, r_norm, r_vir, boxSize)
    end = time.time()
    print('actual time for profiles: ',end-start)
    return sub_medians, sub_medians_r_vir, extrapolated_sub_ids, extrapolated_central_sub_ids
