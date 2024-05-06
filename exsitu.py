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

def fracs_w_mass_bins(basePath, stype, sub_ids, start_snap = 99, random_frac = 1):
    start = time.time()
    h_const = il.groupcat.loadHeader(basePath, start_snap)['HubbleParam']
    
    #check lowest saved parent index table snapshot
    min_snap = funcs.find_file_with_lowest_number('/vera/ptmp/gc/olwitt/' + stype + '/' + basePath[32:39])
    snaps = np.arange(99,min_snap - 1,-1)
    
    n = snaps.size
    
    z = np.zeros(n)

    #necessary offsets, when not every tracer is important:
    insituStarsInSubOffset = tF.exsituStarsInSubOffset(basePath, start_snap)
    file = '/vera/ptmp/gc/olwitt/' + stype + '/' + basePath[32:39] + f'/parent_indices_{start_snap}.hdf5'
    f = h5py.File(file,'r')   

    parent_indices = f[f'snap_{start_snap}/parent_indices'][:,0]
    num_tracers = parent_indices.shape[0]
    del parent_indices
    
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
    
    #trim zeros at the end:
    which_indices = np.trim_zeros(which_indices,'b').astype(int)
    
    #compute correct offsets:
    ## states, which indices correspond to which subhalo from sub_ids
    help_offsets = np.cumsum(help_offsets).astype(int)
    help_offsets = np.insert(help_offsets,0,0)
    
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
    mergers = np.full(which_indices.shape[0], -1, dtype = int) # output
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
        nums[i,:,:], gal_comp[i,:,:] = lF.binning(parent_indices, location, isInMP, isInMP_sat, isInCentral, sub_ids, help_offsets,\
                                                  mass_bin1, mass_bin2, mass_bin3, mass_bin4, mass_bin5)
        
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
            # parents undergoing their second (or more) merger
#             merger2_tmp = np.where(np.logical_and(np.logical_and(np.logical_and(isInMP == 0, location != -1), old_isInMP == 1),\
#                                                   mergers >= 0))[0]
#             mergers[merger2_tmp] = 1
            
            #parents undergoing their first merger
#             merger1_tmp = np.where(np.logical_and(np.logical_and(np.logical_and(isInMP == 0, location != -1), old_isInMP == 1),\
#                                                   mergers < 0))[0]
#             mergers[merger1_tmp] = 0
            
            merger_tmp = np.where(np.logical_and(np.logical_and(isInMP == 0, location != -1), old_isInMP == 1))[0]
            first = np.where(mergers[merger_tmp] < 0)[0]
            more = np.where(mergers[merger_tmp] >= 0)[0]
            
            mergers[merger_tmp[first]] = 0
            mergers[merger_tmp[more]] = 1
            del first, more
#             mergers[merger_tmp] = 1
            
            # mark parents, that entered the MP via a merger BEFORE they were accreted smoothly (e.g. due to wind recycling)
            merger_first_tmp = np.where(np.logical_and(smooth == 0, mergers >= 0))[0]
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
    mergers[np.where(stripped_from_halos == 1)] = -1
    
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
    igm_mergers = np.where(np.logical_and(directly_from_igm == 1, mergers >= 0))[0]
    igm_stripped = np.where(np.logical_and(directly_from_igm == 1, stripped_from_halos == 1))[0]
    merger_stripped = np.where(np.logical_and(stripped_from_halos == 1, mergers >= 0))[0]
    
#     print('back to satellites: ', np.nonzero(back_to_sat)[0].shape[0])
    
    print('# of tracers in two categories: ',igm_mergers.shape, igm_stripped.shape, merger_stripped.shape)
    print('apparent # of tracers in categories: ', np.nonzero(directly_from_igm)[0].shape[0] + np.where(mergers >= 0)[0].shape[0] +\
          np.nonzero(stripped_from_halos)[0].shape[0])
    print('actual total # of tracers: ', mergers.shape)
    no_cat = np.where(np.logical_and(np.logical_and(mergers == 0, directly_from_igm == 0),stripped_from_halos == 0))[0]
    print('# of tracers in no category: ', no_cat.shape)   
    
    print('# no cat from other halos: ', np.where(from_other_halos[no_cat] == 1)[0].shape[0])
    print('# no cat mergers: ', np.where(mergers[no_cat] >= 0)[0].shape[0])
    print('# no cat smooth accretion: ', np.where(smooth[no_cat] == 1)[0].shape[0])
#     print(no_cat)
    
    del igm_mergers, igm_stripped, merger_stripped, no_cat, alwaysInMP, smooth#, back_to_sat
    
    # compute accretion channel fractions for every galaxy individually
    gal_accretion_channels = accretion_channels_all_gals(sub_ids, help_offsets, directly_from_igm, from_other_halos, mergers, stripped_from_halos, long_range_wind_recycled, nep_wind_recycled)
    
    # convert arrays into overall fractions (save arrays?)
    directly_from_igm = np.nonzero(directly_from_igm)[0].shape[0] / directly_from_igm.shape[0]
    from_other_halos = np.nonzero(from_other_halos)[0].shape[0] / from_other_halos.shape[0]
    mergers = np.where(mergers >= 0)[0].shape[0] / mergers.shape[0]
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
    
    #determine mass fractions for every galaxy individually
    for i in range(0,help_offsets.shape[0] - 1):
        indices = np.arange(help_offsets[i],help_offsets[i+1])
        
        star_mask = np.where(parent_indices[indices,1] == 1)[0]
        gas_mask = np.where(parent_indices[indices,1] == 0)[0]
        
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
