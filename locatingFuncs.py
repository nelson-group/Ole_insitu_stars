import numpy as np
import h5py
from numba import jit, njit
import numba as nb
import illustris_python as il
import tracerFuncs as tF
import funcs
import locatingFuncs as lF
import illustrisFuncs as iF
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
    
    # assert random_frac > 0 and random_frac <= 1, 'random fraction has to be > 0 and <= 1!'
    #load number of particles per galaxy to avoid classifying parents as bound to a galaxy 
    #while they're in reality "inner fuzz" of a halo
    
    gasNumInSub_target_snap = numInSub_target_snap[:,0].copy()
    starNumInSub_target_snap = numInSub_target_snap[:,4].copy()
    del numInSub_target_snap
    
    #for resolution comparison reasons: only use <random_frac> fraction of all tracers:
    # if random_frac < 1:
    #     rng = np.random.default_rng()
    #     random_parent_indices = np.zeros(parent_indices.shape)
    #     new_help_offsets = np.zeros(help_offsets.shape[0]).astype(int)
    #     for i in range(0, help_offsets.shape[0] - 1):
    #         indices = np.arange(help_offsets[i],help_offsets[i+1])
    #         size = int(indices.size * random_frac)
    #         new_help_offsets[i+1] = size + new_help_offsets[i]
    #         if size > 0:
    #             parent_indices_indices = rng.choice(indices, size, replace = False, shuffle = False).astype(int)
    #             random_parent_indices[new_help_offsets[i]:new_help_offsets[i+1]] =\
    #             parent_indices[np.sort(parent_indices_indices)]
    
    #     help_offsets = new_help_offsets.copy()
    #     not_zero = np.where(random_parent_indices[:,0] != 0)[0]
    #     random_parent_indices = random_parent_indices[not_zero,:]
    #     parent_indices = random_parent_indices.copy()

    #     assert parent_indices.shape[0] == new_help_offsets[-1]
    #     assert new_help_offsets[0] == 0
        
    #     del random_parent_indices, new_help_offsets
        
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
        
        # if target_snap == 99:
        #     assert (np.where(location[parentIndicesInSub] == main_prog)[0].shape[0]) == parentIndicesInSub.shape[0],\
        #     'offsets wrong probably'   
        #     assert np.all(isInMP[parentIndicesInSub] == 1), 'offsets wrong probably'
    
    time_isInMP = time.time()
    return location, isInMP, isInMP_sat, help_offsets, time_locating - start, time_isInMP - time_locating

def save_location(basePath, stype, start_snap = 99):
    start = time.time()

    num_subs = il.groupcat.loadHeader(basePath,start_snap)['Nsubgroups_Total']
    sub_ids = np.arange(num_subs)
    
    #load all MPBs
    tree_ids = loadMPBs(basePath, start_snap, ids = sub_ids, fields=['SubfindID'])

    #necessary offsets, when not every tracer is important:
    if stype == 'insitu':
        insituStarsInSubOffset = tF.insituStarsInSubOffset(basePath, start_snap)
    elif stype == 'exsitu':
        insituStarsInSubOffset = tF.exsituStarsInSubOffset(basePath, start_snap)
    
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

###################################################################################################

###################################################################################################

########## fracs with mass bins (accretion channels) ##########

def fracs_w_mass_bins(basePath, stype, sub_ids, start_snap = 99, random_frac = 1, save_cats = False):
    start = time.time()
    h_const = il.groupcat.loadHeader(basePath, start_snap)['HubbleParam']
    run = basePath[38]
    
    #check lowest saved parent index table snapshot
#     min_snap = funcs.find_file_with_lowest_number('/vera/ptmp/gc/olwitt/' + stype + '/' + basePath[32:39])
    min_snap = 0
    snaps = np.arange(99,min_snap - 1,-1)
    
    n = snaps.size
    
    z = np.zeros(n)

    groups = il.groupcat.loadHalos(basePath, start_snap, fields = ['Group_M_Crit200','GroupFirstSub'])
    group_masses = groups['Group_M_Crit200']*1e10/h_const

    #differentiate between halos of dwarf / milky way / group size
    cluster_ids = np.where(group_masses >= 10**(13.4))[0]
    group_ids = np.where(np.logical_and(group_masses >= 10**(12.6), group_masses < 10**(13.4)))[0]
    mw_ids = np.where(np.logical_and(group_masses >= 10**(11.8), group_masses < 10**(12.2)))[0]
    dwarf_ids = np.where(np.logical_and(group_masses >= 10**(10.8), group_masses < 10**(11.2)))[0]

    #find ids of associated centrals
    # dtype of sub_ids... is int32 -> change to int64
    sub_ids_groups = groups['GroupFirstSub'][group_ids].astype(np.int64)
    sub_ids_mws = groups['GroupFirstSub'][mw_ids].astype(np.int64)
    sub_ids_dwarfs = groups['GroupFirstSub'][dwarf_ids].astype(np.int64)
    sub_ids_clusters = groups['GroupFirstSub'][cluster_ids].astype(np.int64)

    del groups, group_masses, group_ids, mw_ids, dwarf_ids, cluster_ids

    #necessary offsets, when not every tracer is important:
    if stype.lower() in ['insitu','in-situ','in']:
        stype = 'insitu'
        insituStarsInSubOffset = tF.insituStarsInSubOffset(basePath, start_snap)
    elif stype.lower() in ['exsitu','ex-situ','ex']:
        stype = 'exsitu'
        # raise Exception('Not working at the moment! Only \"in-situ\" available.')
        insituStarsInSubOffset = tF.exsituStarsInSubOffset(basePath, start_snap)
    elif stype.lower() in ['dm','darkmatter','dark matter']:
        stype = 'dm'
        file = '/vera/ptmp/gc/olwitt/' + stype + '/' + basePath[32:39] + f'/dm_indices_{start_snap}.hdf5'
        assert isfile(file), 'DM indices file does not exist!'
        f = h5py.File(file,'r')   

        parent_indices = f['dm_indices'][:]
        parentsInSubOffset = f['dmInSubOffset'][:]
        num_tracers = parent_indices.shape[0]
        f.close()
        del parent_indices
    else:
        raise Exception('Invalid star/particle type!')

    if stype != 'dm':    
        file = '/vera/ptmp/gc/olwitt/' + stype + '/' + basePath[32:39] + f'/parent_indices_{start_snap}.hdf5'
        assert isfile(file), 'Parent indices file does not exist!'
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
    
    before_indices = time.time()
    
    # load subhalo flags
    sample_file = '/vera/ptmp/gc/olwitt/auxCats/' + basePath[32:39] + f'/subhaloFlag_' + stype + '.hdf5'
    assert isfile(sample_file), 'sample file does not exist!'
    f = h5py.File(sample_file,'r')
    subhaloFlag = f['subhaloFlag'][:]
    f.close()

    # load tracer information: insitu or medsitu
    if stype == 'insitu':
        f = h5py.File('/vera/ptmp/gc/olwitt/auxCats/' + basePath[32:39] + f'/insitu_or_medsitu_{start_snap}.hdf5','r')
        
        situ_cat = f['stellar_assembly'][:]#[which_indices]
        f.close()
    else:
        situ_cat = np.zeros(1)

    assert isfile('/vera/ptmp/gc/olwitt/' + stype + '/' + basePath[32:39] + '/subhalo_index_table.hdf5'), 'location file does not exist!'
    location_file = h5py.File('/vera/ptmp/gc/olwitt/' + stype + '/' + basePath[32:39] + '/subhalo_index_table.hdf5','r')
    
    # save global quantities:
    #n: snapshots, 3: [all tracers, insitu, medsitu]
    mp = np.zeros((n,3))
    mp_stars = np.zeros((n,3))
    mp_gas = np.zeros((n,3))
    igm = np.zeros((n,3))
    sub = np.zeros((n,3))
    other_centrals = np.zeros((n,3))
    other_satellites = np.zeros((n,3))
    mp_satellites = np.zeros((n,3))
    
    total = np.zeros((n,3))

    # save the same quantities for different halo and baryonic mass bins:
    #n: snapshots, 5: [all galaxies, dwarfs, mws, groups, clusters] 5: galaxy baryonic mass bins, 3: [all tracers, insitu, medsitu], 8: fractions from above
    nums = np.zeros((n,5,5,3,8))
    
    # save the same quantities for all subhalos:
    #n: snapshots, sub_ids.shape[0]: for each galaxy, 3: [all tracers, insitu, medsitu], 8: fractions from above
    gal_comp = np.zeros((n,sub_ids.shape[0], 3,8)) #galaxy composition
    
    #no insitu/med-situ distinction for ex-situ stars or DM
    if stype == 'exsitu':
        mp = mp[:,0]
        mp_stars = mp_stars[:,0]
        mp_gas = mp_gas[:,0]
        igm = igm[:,0]
        sub = sub[:,0]
        other_centrals = other_centrals[:,0]
        other_satellites = other_satellites[:,0]
        mp_satellites = mp_satellites[:,0]
        total = total[:,0]
        nums = nums[:,:,:,0,:]
        gal_comp = gal_comp[:,:,0,:]

    elif stype == 'dm':
        mp = mp[:,0]
        mp_stars = mp_stars[:,0]
        mp_gas = mp_gas[:,0]
        igm = igm[:,0]
        sub = sub[:,0]
        other_centrals = other_centrals[:,0]
        other_satellites = other_satellites[:,0]
        mp_satellites = mp_satellites[:,0]
        total = total[:,0]
        nums = nums[:,:,:,0,:6] # no star/gas progenitors for DM -> no stars/gas in MP categories
        gal_comp = gal_comp[:,:,0,:6]

    # find accretion origin for every single tracer

    directly_from_igm = np.ones(num_tracers, dtype = np.byte) #output
    smooth = np.zeros(num_tracers, dtype = np.byte) # helper array
    long_range_wind_recycled = smooth.copy() # output
    from_other_halos = smooth.copy() # output
    mergers = smooth.copy() # output
    stripped_from_satellites = smooth.copy() # output
#     back_to_sat = smooth.copy() # helper array
    merger_before_smooth = smooth.copy() # helper array
    outside_later1 = smooth.copy() # helper array
    # outside_later2 = smooth.copy() # helper array
    # outside_later3 = smooth.copy() # helper array
    # outside_later4 = smooth.copy() # helper array
    recycled = smooth.copy() # output
    
    
    start_loop = time.time()
    print('time for indices: ',start_loop-before_indices, flush = True)
    print('before loop: ',start_loop-start, flush = True)
    
    for i in range(n): #loop over all snapshots
        if stype != 'dm':
            file = '/vera/ptmp/gc/olwitt/' + stype + '/' + basePath[32:39] + f'/parent_indices_{snaps[i]}.hdf5'
            f = h5py.File(file,'r')      

            parent_indices = f[f'snap_{snaps[i]}/parent_indices'][:,:]
            f.close()
        else:
            file = '/vera/ptmp/gc/olwitt/' + stype + '/' + basePath[32:39] + f'/dm_indices_{snaps[i]}.hdf5'
            f = h5py.File(file,'r')      

            parent_indices = f['dm_indices'][:]
            f.close()
        
        before_locating = time.time()
        
        # if the location file does not contain the current snapshot, skip it (e.g. for TNG50-4, snapshots 0 and 1)
        if not location_file.__contains__(f'snap_{snaps[i]}'):
            continue
        
        #load location of parents from file (but only tracers from subhalos that we are interested in)
        location = location_file[f'snap_{snaps[i]}/location'][:]
        
        # decode new datatype:
        location_type = location_file[f'snap_{snaps[i]}/location_type'][:]
        
        isInMP = np.zeros(num_tracers, dtype = np.ubyte)
        isInCentral = np.zeros(num_tracers, dtype = np.ubyte)
        isInMP_sat = np.zeros(num_tracers, dtype = np.ubyte)
        
        isInMP[np.isin(location_type,np.array([1,2]))] = 1 #in theory also where location_type = 3,4 but there are no tracers with 3,4
        isInCentral[np.isin(location_type,np.array([2,5]))] = 1 #in theory also where location_type = 4,6 but there are no tracers with 4,6
        isInMP_sat[np.isin(location_type,np.array([7]))] = 1 #in theory also where location_type = 3,4,6 but there are no tracers with 3,4,6
        
        #load information about tracer location w.r.t. galaxy boundaries

        inside_galaxy = np.zeros(num_tracers, dtype = np.ubyte)
        in_galaxy_region = np.zeros(num_tracers, dtype = np.ubyte)
        in_very_inner_halo = np.zeros(num_tracers, dtype = np.ubyte)
        in_inner_halo = np.zeros(num_tracers, dtype = np.ubyte)
        in_outer_halo = np.zeros(num_tracers, dtype = np.ubyte)
        outside_halo = np.zeros(num_tracers, dtype = np.ubyte)


        f = h5py.File('/vera/ptmp/gc/olwitt/' + stype + '/' + basePath[32:39] + f'/distance_cats/distance_cats_{snaps[i]}.hdf5','r')
        inside_radius = f['tracers_inside_radius'][:]
        inside_galaxy_tmp = np.where(inside_radius > 99)[0]
        inside_galaxy[inside_galaxy_tmp] = 1
        in_galaxy_region_tmp = np.where(inside_radius % 10 == 4)[0]
        in_galaxy_region[in_galaxy_region_tmp] = 1
        in_very_inner_halo_tmp = np.where(inside_radius % 10 == 3)[0]
        in_very_inner_halo[in_very_inner_halo_tmp] = 1
        in_inner_halo_tmp = np.where(inside_radius % 10 == 2)[0]
        in_inner_halo[in_inner_halo_tmp] = 1
        in_outer_halo_tmp = np.where(inside_radius % 10 == 1)[0]
        in_outer_halo[in_outer_halo_tmp] = 1
        outside_halo_tmp = np.where(inside_radius == 0)[0]
        outside_halo[outside_halo_tmp] = 1
        del inside_radius, inside_galaxy_tmp, in_galaxy_region_tmp, in_very_inner_halo_tmp, in_inner_halo_tmp, in_outer_halo_tmp, outside_halo_tmp
        f.close()
        
        if i==0:
            print(f'{np.where(isInMP == 0)[0].shape[0]} tracers not in MP at z=0')
            print(f'{np.where(location == -1)[0].shape[0]} tracers in the IGM at z=0')
            print(f'{np.count_nonzero(in_galaxy_region)} tracers in the galaxy region at z=0')
        
        if(i==1):
            end_locate = time.time()
            print('total time for locating: ',end_locate-before_locating, flush = True)
        #load baryonic masses
        sub_masses_stars = il.groupcat.loadSubhalos(basePath,snaps[i],fields=['SubhaloMassType'])[:,4] * 1e10/h_const
        sub_masses_gas = il.groupcat.loadSubhalos(basePath,snaps[i],fields=['SubhaloMassType'])[:,0] * 1e10/h_const
        sub_masses = sub_masses_stars + sub_masses_gas
        
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
        if stype == 'insitu':
            nums[i,:,:,:,:], gal_comp[i,:,:,:] = lF.binning(parent_indices, location, isInMP, isInMP_sat, isInCentral, situ_cat,\
                                                                        parentsInSubOffset, mass_bin1, mass_bin2, mass_bin3, mass_bin4,\
                                                        mass_bin5, subhaloFlag, sub_ids_dwarfs, sub_ids_mws, sub_ids_groups, sub_ids_clusters)
            
        elif stype == 'exsitu':
            nums[i,:,:,:], gal_comp[i,:,:] = lF.ex_binning(parent_indices, location, isInMP, isInMP_sat, isInCentral,\
                                                                        parentsInSubOffset, mass_bin1, mass_bin2, mass_bin3, mass_bin4,\
                                                        mass_bin5, subhaloFlag, sub_ids_dwarfs, sub_ids_mws, sub_ids_groups, sub_ids_clusters)
        
        else:
            nums[i,:,:,:], gal_comp[i,:,:] = lF.dm_binning(location, isInMP, isInMP_sat, isInCentral,\
                                                                        parentsInSubOffset, mass_bin1, mass_bin2, mass_bin3, mass_bin4,\
                                                        mass_bin5, subhaloFlag, sub_ids_dwarfs, sub_ids_mws, sub_ids_groups, sub_ids_clusters)
        

        if stype == 'insitu':
            medsitu = np.where(situ_cat == 1)[0]
            insitu = np.where(situ_cat == 0)[0]
            
            for j in range(3):
                if j == 0:
                    sub_indices = np.arange(num_tracers)
                elif j == 1:
                    sub_indices = insitu
                else:
                    sub_indices = medsitu
                if sub_indices.shape[0] == 0:
                    continue
                star_mask = np.where(parent_indices[sub_indices,1] == 1)[0]
                gas_mask = np.where(parent_indices[sub_indices,1] == 0)[0]

                #number of star parents in the MP
                mp_stars[i,j] = np.where(isInMP[star_mask] == 1)[0].shape[0]
                
                #number of gas parents in the MP
                mp_gas[i,j] = np.where(isInMP[gas_mask] == 1)[0].shape[0]
                
                #number of parents in the MP
                mp[i,j] = mp_stars[i,j] + mp_gas[i,j]

                #number of parents in the IGM
                igm[i,j] = np.where(location[sub_indices] == -1)[0].shape[0]

                

                #number of parents in satellites of the MP
                mp_satellites[i,j] = np.where(isInMP_sat[sub_indices] == 1)[0].shape[0]

                #number of parents in satellites of other central galaxies than the MP
                other_satellites[i,j] = np.where(np.logical_and(np.logical_and(location[sub_indices] != -1, isInCentral[sub_indices] == 0),\
                                                            isInMP[sub_indices] == 0))[0].shape[0] - mp_satellites[i,j]

                #number of parents in other central galaxies than the MP (other halos)
                other_centrals[i,j] = np.where(np.logical_and(np.logical_and(location[sub_indices] != -1, isInCentral[sub_indices] == 1),\
                                                            isInMP[sub_indices] == 0))[0].shape[0]

                #number of parents in other galaxies than the MP; satellites + other_centrals = sub
                #everything not in the igm is in a subhalo (or a FoF halo)
                sub[i,j] = np.where(location[sub_indices] != -1)[0].shape[0] - mp[i,j]
                
                total[i,j] = igm[i,j] + sub[i,j] + mp[i,j]
                
            assert mp[i,0] == np.where(isInMP == 1)[0].shape[0], 'MP wrong.'

        elif stype == 'exsitu':
            star_mask = np.where(parent_indices[:,1] == 1)[0]
            gas_mask = np.where(parent_indices[:,1] == 0)[0]

            #number of star parents in the MP
            mp_stars[i] = np.where(isInMP[star_mask] == 1)[0].shape[0]
            
            #number of gas parents in the MP
            mp_gas[i] = np.where(isInMP[gas_mask] == 1)[0].shape[0]
            
            #number of parents in the MP
            mp[i] = mp_stars[i] + mp_gas[i]

            #number of parents in the IGM
            igm[i] = np.where(location == -1)[0].shape[0]            

            #number of parents in satellites of the MP
            mp_satellites[i] = np.where(isInMP_sat == 1)[0].shape[0]

            #number of parents in satellites of other central galaxies than the MP
            other_satellites[i] = np.where(np.logical_and(np.logical_and(location != -1, isInCentral == 0), isInMP == 0))[0].shape[0] - mp_satellites[i]

            #number of parents in other central galaxies than the MP (other halos)
            other_centrals[i] = np.where(np.logical_and(np.logical_and(location != -1, isInCentral == 1), isInMP == 0))[0].shape[0]

            #number of parents in other galaxies than the MP; satellites + other_centrals = sub
            #everything not in the igm is in a subhalo (or a FoF halo)
            sub[i] = np.where(location != -1)[0].shape[0] - mp[i]
            
            total[i] = igm[i] + sub[i] + mp[i]

        else:
            
            #number of parents in the MP
            mp[i] = np.where(isInMP == 1)[0].shape[0]

            #number of parents in the IGM
            igm[i] = np.where(location == -1)[0].shape[0]            

            #number of parents in satellites of the MP
            mp_satellites[i] = np.where(isInMP_sat == 1)[0].shape[0]

            #number of parents in satellites of other central galaxies than the MP
            other_satellites[i] = np.where(np.logical_and(np.logical_and(location != -1, isInCentral == 0), isInMP == 0))[0].shape[0] - mp_satellites[i]

            #number of parents in other central galaxies than the MP (other halos)
            other_centrals[i] = np.where(np.logical_and(np.logical_and(location != -1, isInCentral == 1), isInMP == 0))[0].shape[0]

            #number of parents in other galaxies than the MP; satellites + other_centrals = sub
            #everything not in the igm is in a subhalo (or a FoF halo)
            sub[i] = np.where(location != -1)[0].shape[0] - mp[i]
            
            total[i] = igm[i] + sub[i] + mp[i]

        z[i] = il.groupcat.loadHeader(basePath, snaps[i])['Redshift'] 
        
        
        ##### what fraction of baryonic matter forming in situ stars by z=0 was accreted in a certain way? #####

        # from galaxy region to outside
        # use the one below for all particle motion in the halo
        # outside_later1[np.where(np.logical_or(np.logical_or(np.logical_or(in_inner_halo == 1, in_outer_halo == 1), outside_halo == 1), in_very_inner_halo == 1))[0]] = 1

        # or use the one below for more ralistic recycling, i.e. only particles that were in the galaxy region at some point and are ejected to r>0.25 R200c
        outside_later1[np.where(np.logical_or(np.logical_or(in_inner_halo == 1, in_outer_halo == 1), outside_halo == 1))[0]] = 1

        # # from very inner halo to outside
        # outside_later2[np.where(np.logical_or(np.logical_or(in_inner_halo == 1, in_outer_halo == 1), outside_halo == 1))[0]] = 1

        # # from inner halo to outside
        # outside_later3[np.where(np.logical_or(outside_halo == 1, in_outer_halo == 1))[0]] = 1

        # # from outer halo to outside
        # outside_later4[np.where(outside_halo == 1)[0]] = 1

        if snaps[i] < 99 and snaps[i] > 1: #skip start_snapshot and only go until snapshot 2
            # mark each parent currently in another galaxy than the MP:
            other_gal_tmp = np.where(np.logical_and(location != -1, isInMP == 0))[0] 
            directly_from_igm[other_gal_tmp] = 0 #assume all came from igm, delete those that were in other galaxies
            
            # smooth accretion (was in IGM, now in MP)
            smooth_tmp = np.where(np.logical_and(np.logical_and(old_isInMP == 1, location == -1), isInMP == 0))[0] 
            smooth[smooth_tmp] = 1 #assume none were smoothly accreted, mark those that are
            

            # mark each parent currently in another halo than the one of the MP (= intergalactic transfer, mergers)
            other_halos_tmp = np.where(np.logical_and(np.logical_and(location != -1, isInMP == 0), isInMP_sat == 0))[0] 
            from_other_halos[other_gal_tmp] = 1 #assume none from other halos (or galaxies in general, respectively), mark those that are 
            
            # wind recycling (was in MP, now in IGM, and eventually in MP again (z=0))
            wind_rec_tmp = np.where(np.logical_and(old_location == -1, isInMP == 1))[0] 
            long_range_wind_recycled[wind_rec_tmp] = 1

            # mark parents, that were in the galaxy region at some point and are located at greater distances at the next snapshot
            # recycled_tmp = np.where(np.logical_and(outside_later4 == 1, in_outer_halo == 1))[0]
            # recycled[recycled_tmp] = 1

            # recycled_tmp = np.where(np.logical_and(outside_later3 == 1, in_inner_halo == 1))[0]
            # recycled[recycled_tmp] = 1

            # recycled_tmp = np.where(np.logical_and(outside_later2 == 1, in_very_inner_halo == 1))[0]
            # recycled[recycled_tmp] = 1

            recycled_tmp = np.where(np.logical_and(outside_later1 == 1, in_galaxy_region == 1))[0]
            recycled[recycled_tmp] = 1
            
            # mark each parent, that was bound to another subhalo prior to being in the MP
            # conditions: was in different subhalo (not IGM), next snapshot it is in MP, was inside the galaxy, no stripping from sats before
            # treat true mergers (inside_galaxy = 1) and those from galaxies not considered (inside_galaxy = -1) the same
            merger_tmp = np.where(np.logical_and(np.logical_and(np.logical_and(np.logical_and(isInMP == 0, location != -1), old_isInMP == 1),\
                                                 np.abs(old_inside_galaxy) == 1), stripped_from_satellites == 0))[0]
            mergers[merger_tmp] = 1
            
            # tracers stripped from satellites: essentially mergers, but the tracer is outside the galaxy after the 'merger'
            # condition: no merger before
            stripped_from_satellites_tmp = np.where(np.logical_and(np.logical_and(np.logical_and(np.logical_and(isInMP == 0, location != -1),\
                                                                                  old_isInMP == 1), old_inside_galaxy == 0), mergers == 0))[0]
            stripped_from_satellites[stripped_from_satellites_tmp] = 1
            
            # mark parents, that entered the MP via a merger BEFORE they were accreted smoothly (e.g. due to wind recycling)
            merger_first_tmp = np.where(np.logical_and(smooth == 0, np.logical_or(mergers == 1, stripped_from_satellites == 1)))[0]
            merger_before_smooth[merger_first_tmp] = 1
            
            # mark each parent, that was in the MP but then entered a satellite
#             back_to_sat_tmp = np.where(np.logical_and(isInMP == 1, old_isInMP_sat == 1))[0]
#             back_to_sat[back_to_sat_tmp] = 1
            
        #old_location refers to higher snapshot (later time) than location, as snaps[i] decreases in each loop
        old_location = location.copy() 
        old_isInMP = isInMP.copy()
        old_inside_galaxy = inside_galaxy.copy()
#         old_isInMP_sat = isInMP_sat.copy()
        
        if(i==1):
            end_binning = time.time()
            print('total time for binning: ', end_binning-end_create_bins, flush = True)
            print('total time for first loop: ', end_binning-start_loop, flush = True)
        print(snaps[i], 'done;', end = ' ', flush = True)
    print()
    
    #find all tracers that were in halos at some point AND meet the smooth accretion criterion
    stripped_from_halos = np.zeros(num_tracers, dtype = np.byte)
    stripped_from_halos_inds = np.where(np.logical_and(smooth == 1, from_other_halos == 1))[0]
    stripped_from_halos[stripped_from_halos_inds] = 1
    
    # corrections:
    # drop all recyled tracers that were in another galaxy at some point
    recycled2 = recycled.copy()
    recycled[np.where(directly_from_igm == 0)] = 0
    
    stripped_from_halos[np.where(merger_before_smooth == 1)] = 0
    mergers[np.where(stripped_from_halos == 1)] = 0
    stripped_from_satellites[np.where(stripped_from_halos == 1)] = 0
    alwaysInMP = np.where(np.logical_and(smooth == 0, directly_from_igm == 1))[0]
    
    # non-externally-processed (nep) wind recycling
    nep_wind_recycled = np.zeros(num_tracers, dtype = np.byte)
    nep_wind_recycled_inds = np.where(np.logical_and(directly_from_igm == 1, long_range_wind_recycled == 1))[0]
    nep_wind_recycled[nep_wind_recycled_inds] = 1
    
    del location, old_location, isInMP, old_isInMP, stripped_from_halos_inds, nep_wind_recycled_inds#, which_indices
    
    # tracers in multiple categories?
    igm_mergers = np.where(np.logical_and(directly_from_igm == 1, mergers == 1))[0]
    igm_stripped = np.where(np.logical_and(directly_from_igm == 1, stripped_from_halos == 1))[0]
    merger_stripped = np.where(np.logical_and(stripped_from_halos == 1, mergers == 1))[0]

    print('# of tracers in two categories: ',igm_mergers.shape, igm_stripped.shape, merger_stripped.shape)
    print('apparent # of tracers in categories: ', np.nonzero(directly_from_igm)[0].shape[0] + np.nonzero(mergers)[0].shape[0] +\
          np.nonzero(stripped_from_halos)[0].shape[0] + np.nonzero(stripped_from_satellites)[0].shape[0])
    print('actual total # of tracers: ', mergers.shape)
    no_cat = np.where(np.logical_and(np.logical_and(np.logical_and(mergers == 0, directly_from_igm == 0),stripped_from_halos == 0),\
                                     stripped_from_satellites == 0))[0]
    print('# of tracers in no category: ', no_cat.shape)   
    
    print('# no cat from other halos: ', np.where(from_other_halos[no_cat] == 1)[0].shape[0])
    print('# no cat mergers: ', np.where(mergers[no_cat] == 1)[0].shape[0])
    print('# no cat stripped from satellites: ', np.where(stripped_from_satellites[no_cat] == 1)[0].shape[0])
    print('# no cat smooth accretion: ', np.where(smooth[no_cat] == 1)[0].shape[0])
    
    print('# stripped from satellites: ', np.nonzero(stripped_from_satellites)[0].shape[0])
    print('# mergers: ', np.nonzero(mergers)[0].shape[0])
    both = np.where(np.logical_and(stripped_from_satellites == 1, mergers == 1))[0]
    print('# both: ',both.shape[0])

    print('# nep recycled: ', np.count_nonzero(recycled))
    print('# recycled: ', np.count_nonzero(recycled2))
    print('# wind recycled: ', np.count_nonzero(nep_wind_recycled))
    print('# long range wind recycled: ', np.count_nonzero(long_range_wind_recycled))
    print('# directly from igm: ', np.count_nonzero(directly_from_igm))
    print('# fresh accretion: ', np.count_nonzero(directly_from_igm) - np.count_nonzero(recycled))

    if save_cats:
        '''output:
        
            0: fresh accretion from the IGM'
            1: wind recycled in the MP (non-externally processed [NEP])
            2: clumpy accretion of gas (mergers)
            3: clumpy accretion of gas (stripped from satellites)
            4: stripped/ejected from other halos (smooth accretion onto MP)
        '''
        
        res = np.full(num_tracers, -1, np.byte)
        #fresh accretion = directly from igm - nep wind recycled
        res[np.where(np.logical_and(directly_from_igm == 1, nep_wind_recycled == 0))] = 0
        
        #nep wind recycled (new definition!)
        res[np.nonzero(recycled)] = 1
        
        #mergers
        res[np.nonzero(mergers)] = 2
        
        #stripped from satellites
        res[np.nonzero(stripped_from_satellites)] = 3
        
        #stripped/ejected from halos
        res[np.nonzero(stripped_from_halos)] = 4
        
        file = f'/vera/ptmp/gc/olwitt/auxCats/TNG50-{run}/tracer_accretion_channels_{stype}_{start_snap}.hdf5'
        f = h5py.File(file, 'w')
        f.create_dataset('tracer_accretion_channels', data = res)
        f.close()
    
    del igm_mergers, igm_stripped, merger_stripped, no_cat, alwaysInMP, smooth#, back_to_sat
    
    # compute accretion channel fractions for every galaxy individually
    if stype == 'insitu':
        gal_accretion_channels = accretion_channels_all_gals(situ_cat, parentsInSubOffset, directly_from_igm, from_other_halos, mergers, stripped_from_halos, long_range_wind_recycled, nep_wind_recycled, stripped_from_satellites, recycled)
    else:
        gal_accretion_channels = accretion_channels_all_gals_ex_dm(parentsInSubOffset, directly_from_igm, from_other_halos, mergers, stripped_from_halos, long_range_wind_recycled, nep_wind_recycled, stripped_from_satellites, recycled)
    
    # convert arrays into overall fractions
    directly_from_igm = np.nonzero(directly_from_igm)[0].shape[0] / num_tracers
    from_other_halos = np.nonzero(from_other_halos)[0].shape[0] / num_tracers
    mergers = np.nonzero(mergers)[0].shape[0] / num_tracers
    stripped_from_halos = np.nonzero(stripped_from_halos)[0].shape[0] / num_tracers
    long_range_wind_recycled = np.nonzero(long_range_wind_recycled)[0].shape[0] / num_tracers
    nep_wind_recycled = np.nonzero(nep_wind_recycled)[0].shape[0] / num_tracers
    stripped_from_satellites = np.nonzero(stripped_from_satellites)[0].shape[0] / num_tracers
    
    location_file.close()
    return mp, mp_stars, mp_gas, igm, sub, other_satellites, mp_satellites, other_centrals, total, nums, z, gal_comp, subhaloFlag,\
directly_from_igm, stripped_from_halos, from_other_halos, mergers, long_range_wind_recycled, nep_wind_recycled, stripped_from_satellites,\
gal_accretion_channels

# @njit

def ex_binning(parent_indices, location, isInMP, isInMP_sat, isInCentral, offsets, mass_bin1, mass_bin2, mass_bin3,\
            mass_bin4, mass_bin5, subhaloFlag, dwarf_ids, mw_ids, group_ids, cluster_ids):
    res = np.zeros((5,5,8))
    gal_res = np.zeros((offsets.shape[0] - 1,8))

    #determine mass fractions for every galaxy individually
    for i in range(0,offsets.shape[0] - 1):
        indices = np.arange(offsets[i],offsets[i+1])
        
        if indices.shape[0] == 0:
            continue
        star_mask = np.where(parent_indices[indices,1] == 1)[0]
        gas_mask = np.where(parent_indices[indices,1] == 0)[0]

        #number of parents in the MP
        gal_res[i,0] = np.where(isInMP[indices] == 1)[0].shape[0]
        
        #number of star parents in the MP
        gal_res[i,1] = np.where(isInMP[indices[star_mask]] == 1)[0].shape[0]
        
        #number of gas parents in the MP
        gal_res[i,2] = np.where(isInMP[indices[gas_mask]] == 1)[0].shape[0]

        #number of parents in other galaxies
        gal_res[i,3] = np.where(location[indices] != -1)[0].shape[0] - gal_res[i,0] 
        
        #number of parents in satellites of the MP
        gal_res[i,5] = np.where(isInMP_sat[indices] == 1)[0].shape[0]
        
        #number of parents in satellites of other halos
        gal_res[i,4] = np.where(np.logical_and(np.logical_and(location[indices] != -1, isInCentral[indices] == 0), isInMP[indices] == 0))[0].shape[0] - gal_res[i,5]

        #number of parents in other centrals (halos)
        gal_res[i,6] = np.where(np.logical_and(np.logical_and(location[indices] != -1, isInCentral[indices] == 1), isInMP[indices] == 0))[0].shape[0]

        #total
        gal_res[i,7] = indices.shape[0]
        
        #obtain mass fractions
        gal_res[i,:7] = gal_res[i,:7] / gal_res[i,7] if gal_res[i,7] > 0 else gal_res[i,:7]
    
    del indices
    #determine mass fractions for entire mass bins
    #first: only takegalaxies with flag=1
    dwarf_inds = dwarf_ids[np.nonzero(funcs.isin(dwarf_ids, np.nonzero(subhaloFlag)[0]))[0]]
    mw_inds = mw_ids[np.nonzero(funcs.isin(mw_ids, np.nonzero(subhaloFlag)[0]))[0]]
    group_inds = group_ids[np.nonzero(funcs.isin(group_ids, np.nonzero(subhaloFlag)[0]))[0]]
    cluster_inds = cluster_ids[np.nonzero(funcs.isin(cluster_ids, np.nonzero(subhaloFlag)[0]))[0]]

    #loop over halo mass bins
    for h in range(5):
        #all_good.dtype = int64
        all_good = np.nonzero(subhaloFlag)[0]
        halo_mass_bin = all_good if h == 0 else dwarf_inds if h == 1 else mw_inds if h == 2 else group_inds if h == 3 else cluster_inds

        # find all tracers in each halo mass bin
        # halo bin inds points into the entire tracer array (parent_indices, location, etc.)
        halo_bin_inds = iF.find_tracers_of_subs(halo_mass_bin, offsets)

        for i in range(5):
            mass_bin = mass_bin1 if i==0 else mass_bin2 if i==1 else mass_bin3 if i==2 else mass_bin4 if i==3 else\
            mass_bin5
            
            indices = halo_bin_inds[np.nonzero(funcs.isin(location[halo_bin_inds],mass_bin))[0]]

            if indices.shape[0] == 0:
                continue
            
            star_mask = np.where(parent_indices[indices,1] == 1)[0]
            gas_mask = np.where(parent_indices[indices,1] == 0)[0]

            #number of parents in the MP
            res[h,i,0] = np.where(isInMP[indices] == 1)[0].shape[0]
            
            #number of star parents in the MP
            res[h,i,1] = np.where(isInMP[indices[star_mask]] == 1)[0].shape[0]
            
            #number of gas parents in the MP
            res[h,i,2] = np.where(isInMP[indices[gas_mask]] == 1)[0].shape[0]

            #number of parents in other galaxies
            res[h,i,3] = np.where(location[indices] != -1)[0].shape[0] - res[h,i,0] 
            
            #number of parents in satellites of the MP
            res[h,i,5] = np.where(isInMP_sat[indices] == 1)[0].shape[0]
            
            #number of parents in satellites of other halos
            res[h,i,4] = np.where(np.logical_and(np.logical_and(location[indices] != -1, isInCentral[indices] == 0), isInMP[indices] == 0))[0].shape[0] - res[h,i,5]

            #number of parents in other centrals (halos)
            res[h,i,6] = np.where(np.logical_and(np.logical_and(location[indices] != -1, isInCentral[indices] == 1), isInMP[indices] == 0))[0].shape[0]

            #total -> total in halo mass bin
            res[h,i,7] = halo_mass_bin.shape[0]
            
            #obtain mass fractions
            # res[h,i,:7] = res[h,i,:7] / res[h,i,7] if res[h,i,7] > 0 else res[h,i,:7]
    
    return res, gal_res

def dm_binning(location, isInMP, isInMP_sat, isInCentral, offsets, mass_bin1, mass_bin2, mass_bin3,\
            mass_bin4, mass_bin5, subhaloFlag, dwarf_ids, mw_ids, group_ids, cluster_ids):
    res = np.zeros((5,5,6))
    gal_res = np.zeros((offsets.shape[0] - 1,6))

    #determine mass fractions for every galaxy individually
    for i in range(0,offsets.shape[0] - 1):
        indices = np.arange(offsets[i],offsets[i+1])
        
        if indices.shape[0] == 0:
            continue

        #number of parents in the MP
        gal_res[i,0] = np.where(isInMP[indices] == 1)[0].shape[0]

        #number of parents in other galaxies
        gal_res[i,1] = np.where(location[indices] != -1)[0].shape[0] - gal_res[i,0] 
        
        #number of parents in satellites of the MP
        gal_res[i,3] = np.where(isInMP_sat[indices] == 1)[0].shape[0]
        
        #number of parents in satellites of other halos
        gal_res[i,2] = np.where(np.logical_and(np.logical_and(location[indices] != -1, isInCentral[indices] == 0), isInMP[indices] == 0))[0].shape[0] - gal_res[i,3]

        #number of parents in other centrals (halos)
        gal_res[i,4] = np.where(np.logical_and(np.logical_and(location[indices] != -1, isInCentral[indices] == 1), isInMP[indices] == 0))[0].shape[0]

        #total
        gal_res[i,5] = indices.shape[0]
        
        #obtain mass fractions
        gal_res[i,:5] = gal_res[i,:5] / gal_res[i,5] if gal_res[i,5] > 0 else gal_res[i,:5]
    
    del indices
    #determine mass fractions for entire mass bins
    #first: only takegalaxies with flag=1
    dwarf_inds = dwarf_ids[np.nonzero(funcs.isin(dwarf_ids, np.nonzero(subhaloFlag)[0]))[0]]
    mw_inds = mw_ids[np.nonzero(funcs.isin(mw_ids, np.nonzero(subhaloFlag)[0]))[0]]
    group_inds = group_ids[np.nonzero(funcs.isin(group_ids, np.nonzero(subhaloFlag)[0]))[0]]
    cluster_inds = cluster_ids[np.nonzero(funcs.isin(cluster_ids, np.nonzero(subhaloFlag)[0]))[0]]

    #loop over halo mass bins
    for h in range(5):
        #all_good.dtype = int64
        all_good = np.nonzero(subhaloFlag)[0]
        halo_mass_bin = all_good if h == 0 else dwarf_inds if h == 1 else mw_inds if h == 2 else group_inds if h == 3 else cluster_inds

        # find all tracers in each halo mass bin
        # halo bin inds points into the entire tracer array (parent_indices, location, etc.)
        halo_bin_inds = iF.find_tracers_of_subs(halo_mass_bin, offsets)

        for i in range(5):
            mass_bin = mass_bin1 if i==0 else mass_bin2 if i==1 else mass_bin3 if i==2 else mass_bin4 if i==3 else\
            mass_bin5
            
            indices = halo_bin_inds[np.nonzero(funcs.isin(location[halo_bin_inds],mass_bin))[0]]

            if indices.shape[0] == 0:
                continue

            #number of parents in the MP
            res[h,i,0] = np.where(isInMP[indices] == 1)[0].shape[0]

            #number of parents in other galaxies
            res[h,i,1] = np.where(location[indices] != -1)[0].shape[0] - res[h,i,0] 
            
            #number of parents in satellites of the MP
            res[h,i,3] = np.where(isInMP_sat[indices] == 1)[0].shape[0]
            
            #number of parents in satellites of other halos
            res[h,i,2] = np.where(np.logical_and(np.logical_and(location[indices] != -1, isInCentral[indices] == 0), isInMP[indices] == 0))[0].shape[0] - res[h,i,3]

            #number of parents in other centrals (halos)
            res[h,i,4] = np.where(np.logical_and(np.logical_and(location[indices] != -1, isInCentral[indices] == 1), isInMP[indices] == 0))[0].shape[0]

            #total -> total in halo mass bin
            res[h,i,5] = halo_mass_bin.shape[0]
            
            #obtain mass fractions
            # res[h,i,:5] = res[h,i,:5] / res[h,i,5] if res[h,i,5] > 0 else res[h,i,:5]
    
    return res, gal_res

def binning(parent_indices, location, isInMP, isInMP_sat, isInCentral, situ_cat, offsets, mass_bin1, mass_bin2, mass_bin3,\
            mass_bin4, mass_bin5, subhaloFlag, dwarf_ids, mw_ids, group_ids, cluster_ids):
    res = np.zeros((5,5,3,8))
    gal_res = np.zeros((offsets.shape[0] - 1,3,8))
    
    #determine mass fractions for every galaxy individually
    for i in range(0,offsets.shape[0] - 1):
        indices = np.arange(offsets[i], offsets[i+1])
        medsitu = np.where(situ_cat[indices] == 1)[0]
        insitu = np.where(situ_cat[indices] == 0)[0]
        
        
        for j in range(3):
            if j == 0:
                sub_indices = np.arange(indices.shape[0])
            elif j == 1:
                sub_indices = insitu
            else:
                sub_indices = medsitu
            if sub_indices.shape[0] == 0:
                continue
            star_mask = np.where(parent_indices[indices[sub_indices],1] == 1)[0]
            gas_mask = np.where(parent_indices[indices[sub_indices],1] == 0)[0]

            #number of parents in the MP
            gal_res[i,j,0] = np.where(isInMP[indices[sub_indices]] == 1)[0].shape[0]
            
            #number of star parents in the MP
            gal_res[i,j,1] = np.where(isInMP[indices[star_mask]] == 1)[0].shape[0]
            
            #number of gas parents in the MP
            gal_res[i,j,2] = np.where(isInMP[indices[gas_mask]] == 1)[0].shape[0]

            #number of parents in other galaxies
            gal_res[i,j,3] = np.where(location[indices[sub_indices]] != -1)[0].shape[0] - gal_res[i,j,0] 
            
            #number of parents in satellites of the MP
            gal_res[i,j,5] = np.where(isInMP_sat[indices[sub_indices]] == 1)[0].shape[0]
            
            #number of parents in satellites of other halos
            gal_res[i,j,4] = np.where(np.logical_and(np.logical_and(location[indices[sub_indices]] != -1,\
                                                                    isInCentral[indices[sub_indices]] == 0),\
                                                     isInMP[indices[sub_indices]] == 0))[0].shape[0] - gal_res[i,j,5]

            #number of parents in other centrals (halos)
            gal_res[i,j,6] = np.where(np.logical_and(np.logical_and(location[indices[sub_indices]] != -1,\
                                                                    isInCentral[indices[sub_indices]] == 1),\
                                                     isInMP[indices[sub_indices]] == 0))[0].shape[0]

            #total
            gal_res[i,j,7] = sub_indices.shape[0]
            
            #obtain mass fractions
            gal_res[i,j,:7] = gal_res[i,j,:7] / gal_res[i,j,7] if gal_res[i,j,7] > 0 else gal_res[i,j,:7]
    
    del indices
    #determine mass fractions for entire mass bins
    #first: only takegalaxies with flag=1
    dwarf_inds = dwarf_ids[np.nonzero(funcs.isin(dwarf_ids, np.nonzero(subhaloFlag)[0]))[0]]
    mw_inds = mw_ids[np.nonzero(funcs.isin(mw_ids, np.nonzero(subhaloFlag)[0]))[0]]
    group_inds = group_ids[np.nonzero(funcs.isin(group_ids, np.nonzero(subhaloFlag)[0]))[0]]
    cluster_inds = cluster_ids[np.nonzero(funcs.isin(cluster_ids, np.nonzero(subhaloFlag)[0]))[0]]

    #loop over halo mass bins
    for h in range(5):
        #all_good.dtype = int64
        all_good = np.nonzero(subhaloFlag)[0]
        halo_mass_bin = all_good if h == 0 else dwarf_inds if h == 1 else mw_inds if h == 2 else group_inds if h == 3 else cluster_inds

        # find all tracers in each halo mass bin
        # halo bin inds points into the entire tracer array (parent_indices, location, etc.)
        halo_bin_inds = iF.find_tracers_of_subs(halo_mass_bin, offsets)

        for i in range(5):
            mass_bin = mass_bin1 if i==0 else mass_bin2 if i==1 else mass_bin3 if i==2 else mass_bin4 if i==3 else\
            mass_bin5
            
            indices = halo_bin_inds[np.nonzero(funcs.isin(location[halo_bin_inds],mass_bin))[0]]
            
            medsitu = np.where(situ_cat[indices] == 1)[0]
            insitu = np.where(situ_cat[indices] == 0)[0]
            
            for j in range(3):
                if j == 0:
                    sub_indices = np.arange(indices.shape[0])
                elif j == 1:
                    sub_indices = insitu
                else:
                    sub_indices = medsitu
                if sub_indices.shape[0] == 0:
                    continue
            
                star_mask = np.where(parent_indices[indices[sub_indices],1] == 1)[0]
                gas_mask = np.where(parent_indices[indices[sub_indices],1] == 0)[0]

                #number of parents in the MP
                res[h,i,j,0] = np.where(isInMP[indices[sub_indices]] == 1)[0].shape[0]
                
                #number of star parents in the MP
                res[h,i,j,1] = np.where(isInMP[indices[star_mask]] == 1)[0].shape[0]
                
                #number of gas parents in the MP
                res[h,i,j,2] = np.where(isInMP[indices[gas_mask]] == 1)[0].shape[0]

                #number of parents in other galaxies
                res[h,i,j,3] = np.where(location[indices[sub_indices]] != -1)[0].shape[0] - res[h,i,j,0] 
                
                #number of parents in satellites of the MP
                res[h,i,j,5] = np.where(isInMP_sat[indices[sub_indices]] == 1)[0].shape[0]
                
                #number of parents in satellites of other halos
                res[h,i,j,4] = np.where(np.logical_and(np.logical_and(location[indices[sub_indices]] != -1,\
                                                                        isInCentral[indices[sub_indices]] == 0),\
                                                        isInMP[indices[sub_indices]] == 0))[0].shape[0] - res[h,i,j,5]

                #number of parents in other centrals (halos)
                res[h,i,j,6] = np.where(np.logical_and(np.logical_and(location[indices[sub_indices]] != -1,\
                                                                        isInCentral[indices[sub_indices]] == 1),\
                                                        isInMP[indices[sub_indices]] == 0))[0].shape[0]

                #total -> total in halo mass bin
                res[h,i,j,7] = halo_mass_bin.shape[0]
                
                #obtain mass fractions
                # res[h,i,j,:7] = res[h,i,j,:7] / res[h,i,j,7] if res[h,i,j,7] > 0 else res[h,i,j,:7]
    
    return res, gal_res

@jit(nopython = True, parallel = True)
def accretion_channels_all_gals(situ_cat, offsets, directly_from_igm, from_other_halos, mergers, stripped_from_halos, long_range_wind_recycled, nep_wind_recycled, stripped_from_satellites, recycled):
    res = np.zeros((offsets.shape[0] - 1,3,8))
    for i in range(offsets.shape[0] - 1):
        indices = np.arange(offsets[i], offsets[i+1])
        
        medsitu = np.where(situ_cat[indices] == 1)[0]
        insitu = np.where(situ_cat[indices] == 0)[0]
        
        for j in range(3):
            if j == 0:
                sub_indices = np.arange(indices.shape[0])
            elif j == 1:
                sub_indices = insitu
            else:
                sub_indices = medsitu
            if sub_indices.shape[0] == 0:
                continue
        
            res[i,j,0] = np.nonzero(directly_from_igm[indices[sub_indices]])[0].shape[0] / sub_indices.shape[0]
            res[i,j,1] = np.nonzero(from_other_halos[indices[sub_indices]])[0].shape[0] / sub_indices.shape[0]
            res[i,j,2] = np.nonzero(mergers[indices[sub_indices]])[0].shape[0] / sub_indices.shape[0]
            res[i,j,3] = np.nonzero(stripped_from_halos[indices[sub_indices]])[0].shape[0] / sub_indices.shape[0]
            res[i,j,4] = np.nonzero(long_range_wind_recycled[indices[sub_indices]])[0].shape[0] / sub_indices.shape[0]
            res[i,j,5] = np.nonzero(nep_wind_recycled[indices[sub_indices]])[0].shape[0] / sub_indices.shape[0]
            res[i,j,6] = np.nonzero(stripped_from_satellites[indices[sub_indices]])[0].shape[0] / sub_indices.shape[0]
            res[i,j,7] = np.count_nonzero(recycled[indices[sub_indices]]) / sub_indices.shape[0]
        
    return res

@jit(nopython = True, parallel = True)
def accretion_channels_all_gals_ex_dm(offsets, directly_from_igm, from_other_halos, mergers, stripped_from_halos, long_range_wind_recycled, nep_wind_recycled, stripped_from_satellites, recycled):
    res = np.zeros((offsets.shape[0] - 1,8))
    for i in range(offsets.shape[0] - 1):
        indices = np.arange(offsets[i], offsets[i+1])

        if indices.shape[0] == 0:
            continue
        
        res[i,0] = np.nonzero(directly_from_igm[indices])[0].shape[0] / indices.shape[0]
        res[i,1] = np.nonzero(from_other_halos[indices])[0].shape[0] / indices.shape[0]
        res[i,2] = np.nonzero(mergers[indices])[0].shape[0] / indices.shape[0]
        res[i,3] = np.nonzero(stripped_from_halos[indices])[0].shape[0] / indices.shape[0]
        res[i,4] = np.nonzero(long_range_wind_recycled[indices])[0].shape[0] / indices.shape[0]
        res[i,5] = np.nonzero(nep_wind_recycled[indices])[0].shape[0] / indices.shape[0]
        res[i,6] = np.nonzero(stripped_from_satellites[indices])[0].shape[0] / indices.shape[0]
        res[i,7] = np.count_nonzero(recycled[indices]) / indices.shape[0]
        
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