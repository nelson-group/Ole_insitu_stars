import time
import illustris_python as il
import numpy as np
import h5py
import numba as nb
from numba import jit, njit
import tracerFuncs as tF
import funcs
from os.path import isfile, isdir
import sys
import psutil

sys.path.append('/vera/u/olwitt/illustris_python/illustris_python')
from loadMPBs import loadMPBs

@jit(nopython = True)
def distance_binning(distance, numBins, max_dist): 
    minVal = 0
    maxVal = max_dist
    
    binWidth = (maxVal - minVal) / numBins
    
    yMed = np.zeros(numBins)
    
    for j in range(numBins):
        relInd = np.where(np.logical_and(distance >= minVal + j*binWidth, distance < minVal + (j+1)*binWidth))[0]
        if(relInd.shape[0] > 0):
            yMed[j] = relInd.shape[0] / distance.shape[0]
            
    return yMed

@jit(nopython = True, parallel = True)
def isSubGalaxy(sub_ids, final_offsets):
    """ Checks the number of tracers in each galaxy and marks the ones with 0 tracers."""
    
    noGalaxy = np.ones(sub_ids.shape[0], dtype = np.ubyte)
    for i in nb.prange(sub_ids.shape[0]):
        sub_id = sub_ids[i]
        if final_offsets[sub_id + 1] - final_offsets[sub_id] == 0:
            continue
        noGalaxy[i] = 0
    return noGalaxy

@jit(nopython = True)#, parallel = True)
def distances(parent_indices_data, final_offsets, all_gas_pos, all_star_pos, sub_pos_at_target_snap, subhaloFlag, sub_ids,\
               shmr, r_vir, boxSize, star_formation_snaps, target_snap, shmr_cut, r_vir_cut, accretion_channels, situ_cat,\
                  return_profiles, num_hmr, num_r_vir, num_bins):
    """ For each galaxy with >1 tracers, the distance of every tracer to the (MP) subhalo center is computed.
      Furthermore, it checks whether the tracers is located within the halo or even the galaxy."""
    
    sub_medians = np.full((sub_ids.shape[0],3,3),np.nan, dtype = np.float32)
    sub_medians_r_vir = np.full((sub_ids.shape[0],3,3),np.nan, dtype = np.float32)
    
    if return_profiles:
        profiles_hmr = np.full((sub_ids.shape[0],3,num_bins), np.nan, dtype = np.float32)
        profiles_r_vir = np.full((sub_ids.shape[0],3,num_bins), np.nan, dtype = np.float32)
        
        profiles_situ_hmr = np.full((sub_ids.shape[0],3,num_bins), np.nan, dtype = np.float32)
        profiles_situ_r_vir = np.full((sub_ids.shape[0],3,num_bins), np.nan, dtype = np.float32)

        # profiles_hmr = np.zeros((1,1,1), dtype = np.float32)
        # profiles_r_vir = np.full((sub_ids.shape[0],num_bins), np.nan, dtype = np.float32)
        
        # profiles_situ_hmr = np.zeros((1,1,1), dtype = np.float32)
        # profiles_situ_r_vir = np.zeros((1,1,1), dtype = np.float32)
    else:
        profiles_hmr = np.zeros((1,1,1), dtype = np.float32)
        profiles_r_vir = np.zeros((1,1,1), dtype = np.float32)
        
        profiles_situ_hmr = np.zeros((1,1,1), dtype = np.float32)
        profiles_situ_r_vir = np.zeros((1,1,1), dtype = np.float32)
    
    #all tracers from galaxies that are no longer centrals are marked as -1
    #all tracers outside of galaxies (which are still centals) are marked as 0
    #all tracers inside of galaxies (which are still centals) are marked as 1
    
    inside_2shmr = np.full(parent_indices_data.shape[0], -1, dtype = np.byte)
    inside_r_vir = np.full(parent_indices_data.shape[0], -1, dtype = np.byte)
    
    num_new_stars = np.where(star_formation_snaps == target_snap)[0].shape[0]
    dist_at_star_form = np.full(num_new_stars, -1, dtype = np.float32)
    
    #compute offsets so the second loop can run in parallel
    
    star_form_offsets = np.zeros(sub_ids.shape[0], dtype = np.int32)
    for i in nb.prange(sub_ids.shape[0]):
        indices_of_sub = np.arange(final_offsets[sub_ids[i]],final_offsets[sub_ids[i]+1])
        new_stars_in_sub = np.where(star_formation_snaps[indices_of_sub] == target_snap)[0]
        num_new_stars_in_sub = new_stars_in_sub.shape[0]
        star_form_offsets[i] = num_new_stars_in_sub
        
    star_form_offsets = np.cumsum(star_form_offsets)
    #numba compatible insert function:
    star_form_offsets = funcs.insert(star_form_offsets, 0, 0)
#     star_form_offsets = np.insert(star_form_offsets, 0, 0)

    # the following assertion is not fulfilled as we are only considering centrals.
    # star_formation_snaps includes tracers of every subhalo
#     assert star_form_offsets[-1] == num_new_stars
        
    for i in nb.prange(sub_ids.shape[0]):
        
        #skip unsuitable subhalos 
        if subhaloFlag[i] == 0:
            continue
            
        sub_id = sub_ids[i]
        indices_of_sub = np.arange(final_offsets[sub_id],final_offsets[sub_id+1])
        ac_ch_at_cut = accretion_channels[indices_of_sub]
        
        parent_indices_of_sub = parent_indices_data[indices_of_sub,:]

        particle_pos = np.zeros((indices_of_sub.shape[0],3))
        gas_mask = np.where(parent_indices_of_sub[:,1] == 0)[0]
        star_mask = np.where(parent_indices_of_sub[:,1] == 1)[0]
        
        gas_parent_indices = parent_indices_of_sub[gas_mask,0]
        particle_pos[gas_mask,:] = all_gas_pos[gas_parent_indices,:]

        star_parent_indices = parent_indices_of_sub[star_mask,0]
        particle_pos[star_mask,:] = all_star_pos[star_parent_indices,:]
        
        subhalo_position = sub_pos_at_target_snap[sub_id,:] 

        rad_dist = funcs.dist_vector_nb(subhalo_position,particle_pos,boxSize)
        
        #radius crossings:
        
        in_gal = np.where(rad_dist < shmr_cut * shmr[i])[0]
        not_in_gal = np.where(rad_dist >= shmr_cut * shmr[i])[0]
        inside_2shmr[indices_of_sub[in_gal]] = 1
        inside_2shmr[indices_of_sub[not_in_gal]] = 0
        
        in_halo = np.where(rad_dist < r_vir_cut * r_vir[i])[0]
        not_in_halo = np.where(rad_dist >= r_vir_cut * r_vir[i])[0]
        inside_r_vir[indices_of_sub[in_halo]] = 1
        inside_r_vir[indices_of_sub[not_in_halo]] = 0
        
        #radius at star formation (normalized by shmr):
        
        new_stars_in_sub = np.where(star_formation_snaps[indices_of_sub] == target_snap)[0]
        num_new_stars_in_sub = new_stars_in_sub.shape[0]
        dist_at_star_form[star_form_offsets[i]:star_form_offsets[i+1]] = rad_dist[new_stars_in_sub] / shmr[i]
        
        #igm_mask: all tracers directly from the igm, i.e. from fresh accretion ('0') or nep wind recycling ('1')
        #satellite_mask: all tracers that entered via mergers ('2')
        igm_mask = np.where(funcs.isin(ac_ch_at_cut,np.array([0,1])))[0]
        satellite_mask = np.where(ac_ch_at_cut == 2)[0]
        
        
        #radial profiles:
        if return_profiles:
            
            for j in range(3):
                if j == 0:
                    subset = np.arange(indices_of_sub.shape[0])
                    subset2 = subset
                elif j == 1:
                    #insitu
                    subset = np.where(situ_cat[indices_of_sub] == 0)[0]
                    subset2 = igm_mask
                    
                else:
                    #medsitu
                    subset = np.where(situ_cat[indices_of_sub] == 1)[0]
                    subset2 = satellite_mask
                
                max_dist = num_hmr * shmr[i]
                if subset.shape[0] > 0:
                    # pass
                    profiles_situ_hmr[i,j,:] = distance_binning(rad_dist[subset],num_bins, max_dist)
                if subset2.shape[0] > 0:
                    # pass    
                    profiles_hmr[i,j,:] = distance_binning(rad_dist[subset2],num_bins, max_dist)

                max_dist = num_r_vir * r_vir[i]
                if subset.shape[0] > 0:
                    # pass
                    profiles_situ_r_vir[i,j,:] = distance_binning(rad_dist[subset],num_bins, max_dist)
                if subset2.shape[0] > 0 and j == 0: #added second statement
                    profiles_r_vir[i,:] = distance_binning(rad_dist[subset2],num_bins, max_dist) #change: middle index j deleted
        
        #Lagrangian region computations:
        for j in range(3):
            if j == 0:
                subset = np.arange(indices_of_sub.shape[0])
            elif j == 1:
                #insitu
                subset = np.where(situ_cat[indices_of_sub] == 0)[0]
            else:
                #medsitu
                subset = np.where(situ_cat[indices_of_sub] == 1)[0]

            if subset.shape[0] == 0:
                continue
                
            subset_igm_mask = np.nonzero(funcs.isin(subset,igm_mask))[0]
            subset_satellite_mask = np.nonzero(funcs.isin(subset,satellite_mask))[0]
            
            sub_medians[i,j,0] = np.median(rad_dist[subset]) / shmr[i]
            sub_medians_r_vir[i,j,0] = np.median(rad_dist[subset]) / r_vir[i]
            
            if subset_igm_mask.size > 0:
                sub_medians[i,j,1] = np.median(rad_dist[subset_igm_mask]) / shmr[i]
                sub_medians_r_vir[i,j,1] = np.median(rad_dist[subset_igm_mask]) / r_vir[i]

            if subset_satellite_mask.size > 0:
                sub_medians[i,j,2] = np.median(rad_dist[subset_satellite_mask]) / shmr[i]
                sub_medians_r_vir[i,j,2] = np.median(rad_dist[subset_satellite_mask]) / r_vir[i]
        
    return sub_medians, sub_medians_r_vir, inside_2shmr, inside_r_vir, dist_at_star_form, profiles_hmr, profiles_r_vir,\
          profiles_situ_hmr, profiles_situ_r_vir

@jit(nopython = True, parallel = True)
def distances_dm(parent_indices_data, final_offsets, all_dm_pos, sub_pos_at_target_snap, subhaloFlag, sub_ids, shmr, r_vir,\
                  boxSize, shmr_cut, r_vir_cut, return_profiles, num_hmr, num_r_vir, num_bins):
    """ For each galaxy with >1 tracers, the distance of every tracer to the (MP) subhalo center is computed.
      Furthermore, it checks whether the tracers is located within the halo or even the galaxy."""
    
    sub_medians = np.full(sub_ids.shape[0], np.nan, dtype = np.float32)
    sub_medians_r_vir = np.full(sub_ids.shape[0], np.nan, dtype = np.float32)
    
    inside_2shmr = np.full(parent_indices_data.shape[0], -1, dtype = np.byte)
    inside_r_vir = np.full(parent_indices_data.shape[0], -1, dtype = np.byte)
        
    if return_profiles:
        profiles_hmr = np.full((sub_ids.shape[0],num_bins), np.nan, dtype = np.float32)
        profiles_r_vir = np.full((sub_ids.shape[0],num_bins), np.nan, dtype = np.float32)
    else:
        profiles_hmr = np.zeros((1,1,1), dtype = np.float32)
        profiles_r_vir = np.zeros((1,1,1), dtype = np.float32)
    
    for i in nb.prange(sub_ids.shape[0]):
        
        #skip unsuitable subhalos 
        if subhaloFlag[i] == 0:
            continue
            
        sub_id = sub_ids[i]
        indices_of_sub = np.arange(final_offsets[sub_id],final_offsets[sub_id+1])
        dm_indices_of_sub = parent_indices_data[indices_of_sub]
        particle_pos = all_dm_pos[dm_indices_of_sub,:]
        
        subhalo_position = sub_pos_at_target_snap[sub_id,:]

        rad_dist = funcs.dist_vector_nb(subhalo_position,particle_pos,boxSize)
        
        #radius crossings:
        
        in_gal = np.where(rad_dist < shmr_cut * shmr[i])[0]
        not_in_gal = np.where(rad_dist >= shmr_cut * shmr[i])[0]
        inside_2shmr[indices_of_sub[in_gal]] = 1
        inside_2shmr[indices_of_sub[not_in_gal]] = 0
        
        in_halo = np.where(rad_dist < r_vir_cut * r_vir[i])[0]
        not_in_halo = np.where(rad_dist >= r_vir_cut * r_vir[i])[0]
        inside_r_vir[indices_of_sub[in_halo]] = 1
        inside_r_vir[indices_of_sub[not_in_halo]] = 0
        
        #radial profiles:
        
        if return_profiles:
            max_dist = num_hmr * shmr[i]
            profiles_hmr[i,:] = distance_binning(rad_dist, num_bins, max_dist)
            
            max_dist = num_r_vir * r_vir[i]
            profiles_r_vir[i,:] = distance_binning(rad_dist, num_bins, max_dist)
        
        #Lagrangian region computations:
                
        if rad_dist.size > 0:
            sub_medians[i] = np.median(rad_dist) / shmr[i]
            
        if rad_dist.size > 0:
            sub_medians_r_vir[i] = np.median(rad_dist) / r_vir[i]
    
    return sub_medians, sub_medians_r_vir, inside_2shmr, inside_r_vir, profiles_hmr, profiles_r_vir

def lagrangian_region(basePath, stype, start_snap, target_snap, shmr_cut, r_vir_cut, use_sfr_gas_hmr = False,\
                      return_profiles = False, num_hmr = None, num_r_vir = None, numBins = None, cumulative = True):
    start_loading = time.time()
    header = il.groupcat.loadHeader(basePath,target_snap)
    h_const = header['HubbleParam']
    boxSize = header['BoxSize']
    num_subs = il.groupcat.loadHeader(basePath,start_snap)['Nsubgroups_Total']
    run = basePath[38]

    #introduce mass bins (just for analysis, not for computation):
    groups = il.groupcat.loadHalos(basePath, start_snap, fields = ['Group_M_Crit200','GroupFirstSub'])
    group_masses = groups['Group_M_Crit200']*1e10/h_const

    #differentiate between halos of dwarf / milky way / group size
    dwarf_ids = np.where(np.logical_and(group_masses > 10**(10.8), group_masses < 10**(11.2)))[0]
    mw_ids = np.where(np.logical_and(group_masses > 10**(11.8), group_masses < 10**(12.2)))[0]
    group_ids = np.where(np.logical_and(group_masses > 10**(12.6), group_masses < 10**(13.4)))[0]
    giant_ids = np.where(group_masses > 10**(13.4))[0]

    #find ids of associated centrals
    sub_ids_dwarfs = groups['GroupFirstSub'][dwarf_ids]
    sub_ids_mw = groups['GroupFirstSub'][mw_ids]
    sub_ids_groups = groups['GroupFirstSub'][group_ids]
    sub_ids_giants = groups['GroupFirstSub'][giant_ids]
    central_ids = groups['GroupFirstSub'][:]
    
#     dwarf_inds = tF.getIndices(sub_ids_dwarfs, all_central_ids)
#     mw_inds = tF.getIndices(sub_ids_mw, all_central_ids)
#     group_inds = tF.getIndices(sub_ids_groups, all_central_ids)
#     giant_inds = tF.getIndices(sub_ids_giants, all_central_ids)
    
    del groups, group_masses, dwarf_ids, mw_ids, group_ids, giant_ids

    sub_ids = np.arange(num_subs)

    #Filter out halos without any subhalo and satellites
    subhaloFlag = np.zeros(num_subs, dtype = np.ubyte)
    subhaloFlag[central_ids[np.where(central_ids != -1)]] = 1
    
    #load MPB trees
    trees = loadMPBs(basePath, start_snap, ids = sub_ids, fields = ['SubfindID'])
    
    #load data from files ---------------------------------------------------------------------------------
    sub_positions = h5py.File('files/'+basePath[32:39]+'/SubhaloPos_new_extrapolated.hdf5','r') 
    is_extrapolated = sub_positions['is_extrapolated'][:]
    
    #load subhalo positions (99 instead of start_snap as they were computed for start_snap = 99)
    sub_pos_at_target_snap = sub_positions['SubhaloPos'][:,99-target_snap,:]
    
    sub_positions.close()
    
    if stype.lower() in ['insitu', 'exsitu']:
        file = '/vera/ptmp/gc/olwitt/' + stype + '/' + basePath[32:39] + f'/parent_indices_{target_snap}.hdf5'
        f = h5py.File(file,'r')

        parent_indices = f[f'snap_{target_snap}/parent_indices'][:,:]
        parent_indices_data = parent_indices[:,:].astype(int)
    
    #offsets -----------------------------------------------------------------------------------------------
        #here, it's okay that the offsets at the target snapshot are used as they are identical at every snapshot
        if f.__contains__(f'snap_{target_snap}/numTracersInParents'):
            numTracersInParents = f[f'snap_{target_snap}/numTracersInParents'][:]
        else:
            numTracersInParents = f[f'snap_{target_snap}/tracers_in_parents_offset'][:]
        f.close()
        
        if stype == 'insitu':
            insituStarsInSubOffset = tF.insituStarsInSubOffset(basePath,start_snap)
        else:
            insituStarsInSubOffset = tF.exsituStarsInSubOffset(basePath,start_snap)
            
        final_offsets = tF.tracersInSubhalo(insituStarsInSubOffset,numTracersInParents).astype(int)
        final_offsets = np.insert(final_offsets,0,0)
        
        del insituStarsInSubOffset, numTracersInParents
        
        #load accretion channels for tracers
        file = '/vera/ptmp/gc/olwitt/auxCats/' + basePath[32:39] + f'/tracer_accretion_channels_{start_snap}.hdf5'
        f = h5py.File(file,'r')
        accretion_channels = f['tracer_accretion_channels'][:]
        f.close()
        
    elif stype.lower() in ['dm', 'darkmatter', 'dark_matter', 'dark matter']:
        stype = 'dm'
        
        file = '/vera/ptmp/gc/olwitt/' + stype + '/' + basePath[32:39] + f'/dm_indices_{target_snap}.hdf5'
        f = h5py.File(file,'r')

        parent_indices = f[f'dm_indices'][:]
        parent_indices_data = parent_indices.astype(int)
        
        final_offsets = f['dmInSubOffset'][:]
        f.close()
        
        accretion_channels = None
        
    else:
        raise Exception('Invalid star/particle type!')
    # ^ offsets for the parent index table, that's why at snapshot 99
    
    
    
    #which galaxies? ----------------------------------------------------------------------------------------
    not_extrapolated = np.nonzero(np.logical_not(is_extrapolated))[0]
    subhaloFlag[not_extrapolated] = 0
    
    del is_extrapolated, not_extrapolated
    #test, which galaxies have zero tracers of insitu stars
    # (this already excludes all galaxies without any stars, since they can't have insitu stars)    
    noGalaxy = isSubGalaxy(sub_ids, final_offsets)
    
    #only use galaxies that have at least one tracer particle (at z=0) AND have an extrapolated SubhaloPos entry
    #all galaxies without extrapolated sub_pos history or only 1 tracer: -1
    subhaloFlag[np.nonzero(noGalaxy)[0]] = 0
    print('# of galaxies with 0 tracers: ', np.nonzero(noGalaxy)[0].shape[0])
    del noGalaxy
    
    #now aquire the correct virial radii (consider only those galaxies that are still centrals):
    #-2 bc. GroupFirstSub could contain -1's
    shmr = np.zeros(num_subs, dtype = np.float32)
    target_sub_ids = np.full(num_subs, -2, dtype = np.int32)
    r_vir = np.zeros(num_subs, dtype = np.float32)
    
    #tree check: find missing trees:
    missing = []
    counter = 0
    tree_check = list(trees)
    for i in range(num_subs):
        if i != tree_check[counter]:
            missing.append(i)
            i += 1
            continue
        counter += 1
        
    for i in range(sub_ids.shape[0]):
        if sub_ids[i] in missing or sub_ids[i] >= num_subs:
            subhaloFlag[i] = 0
            
    #<until here, subhaloFlag is identical for every snapshot>
    
    #only load subfindIDs of subhalos with Flag=1
    for i in range(sub_ids.shape[0]):
        #if tree has sufficient entries
        if subhaloFlag[i] == 1 and start_snap - target_snap < trees[sub_ids[i]]['count']:
#             shmr[i] = trees[sub_ids[i]]['SubhaloHalfmassRadType'][start_snap - target_snap][4]
            target_sub_ids[i] = trees[sub_ids[i]]['SubfindID'][start_snap - target_snap]
            
    del trees
    
    #mark all galaxies which aren't centrals anymore
    groupFirstSub = il.groupcat.loadHalos(basePath, target_snap, fields = ['GroupFirstSub'])
    
    #check first whether there are any halos at all
    if isinstance(groupFirstSub, dict):
        print(f'No groups at snapshot {target_snap}! -> return = 14*[-1]')
        return 14*(np.array([-1]),)
        
    central_sub_ids_at_target_snap, GFS_inds, TSID_inds = np.intersect1d(groupFirstSub, target_sub_ids, return_indices = True)
    r_vir_cat = il.groupcat.loadHalos(basePath, target_snap, fields = ['Group_R_Crit200'])[GFS_inds]
    r_vir[TSID_inds] = r_vir_cat
    shmr_cat = il.groupcat.loadSubhalos(basePath, target_snap, fields =\
                                         ['SubhaloHalfmassRadType'])[central_sub_ids_at_target_snap,4]
    
    #activate if necessary:
    sfr_gas_cat = f'/vera/ptmp/gc/olwitt/auxCats/TNG50-{run}/SubhaloHalfmassRad_Gas_Sfr_{target_snap}.hdf5'
    f = h5py.File(sfr_gas_cat, 'r')
    sfr_gas_hmr = f['SubhaloHalfmassRad_Gas_Sfr'][:]
    
    #Flag = 0 indicates no starforming gas cells in that halo
    sfr_gas_hmr_subhaloFlag = f['subhaloFlag'][:]
    f.close()
    
    sfr_gas_hmr_cat = sfr_gas_hmr[central_sub_ids_at_target_snap]
    
    # either use stellar halfmass radii oder starforming gas halfmass radii
    # there will be very few centrals at z=0 that are not centrals at earlier times
    # -> TSID_inds can be used safely
    shmr[TSID_inds] = shmr_cat
    if use_sfr_gas_hmr:
        # galaxies that have starforming gas cells
        sfr = np.nonzero(sfr_gas_hmr_subhaloFlag[central_sub_ids_at_target_snap])[0]
        # galaxies without starforming gas cells
        no_sfr = np.where(sfr_gas_hmr_subhaloFlag[central_sub_ids_at_target_snap] == 0)[0]
        shmr[TSID_inds[sfr]] = sfr_gas_hmr_cat[sfr]
        # special treatment for galaxies without starforming gas cells: use 4 times the stellar halfmass radius
        # to still include them in the analysis -> be careful as strange behaviour for these galaxies might occur
        shmr[TSID_inds[no_sfr]] = shmr_cat[no_sfr] * 2
    
    #only keep subhalos that are still centrals 
    # -> basically every central at z=0 is also a central at earlier times (until the formation
    #snapshot)

    print('number of galaxies: ', np.where(subhaloFlag == 1)[0].shape[0])
    print('number of centrals: ', TSID_inds.shape[0])
    print('number of galaxies without starforming gas cells: ', no_sfr.shape[0])

    mask = np.full(sub_ids.shape[0], True)
    mask[TSID_inds] = False
    subhaloFlag[mask] = 0 
    
    # the masking below is necessary to avoid division by zero
    zero_shmr = np.where(shmr == 0)[0]
    #-> filters out all galaxies without stars (non-sfr subs are already taken care of by using the stellar halfmass radius)

    zero_r_vir = np.where(r_vir == 0)[0]
    print('zero shmr (non-sfr): ', np.where(shmr[TSID_inds[no_sfr]] == 0)[0].shape[0])
    print('zero shmr (sfr): ', np.where(shmr[TSID_inds[sfr]] == 0)[0].shape[0])
    print('zero r_vir: ', np.where(r_vir[TSID_inds] == 0)[0].shape[0])
    subhaloFlag[zero_shmr] = 0
    subhaloFlag[zero_r_vir] = 0

    print('number of galaxies: ', np.where(subhaloFlag == 1)[0].shape[0])

    del TSID_inds, GFS_inds, central_sub_ids_at_target_snap, r_vir_cat, shmr_cat, sfr_gas_hmr_cat, sfr_gas_hmr_subhaloFlag

    mid = time.time()
    print('time for loading and shit: ',mid-start_loading)

    # get star formation snapshot for all tracers

    if stype in ['insitu', 'exsitu']:
        f = h5py.File('/vera/ptmp/gc/olwitt/' + stype + '/' + basePath[32:39] + '/star_formation_snapshots.hdf5','r')
        star_formation_snaps = f['star_formation_snapshot'][:]
        f.close()
        
        if stype == 'insitu':
            file = f'/vera/ptmp/gc/olwitt/auxCats/TNG50-{run}/insitu_or_medsitu_{start_snap}.hdf5'
            f = h5py.File(file,'r')
            #0: insitu, 1: medsitu
            situ_cat = f['stellar_assembly'][:]
            f.close()
        else:
            situ_cat = np.zeros(1, dtype = np.ubyte)
        
        #load particle positions only right before computation begins
        all_gas_pos = il.snapshot.loadSubset(basePath,target_snap, 'gas', fields = ['Coordinates'])
        all_star_pos = il.snapshot.loadSubset(basePath,target_snap, 'stars', fields = ['Coordinates'])
    
        if isinstance(all_star_pos, dict):
            all_star_pos = np.zeros((1,3))
            
        start = time.time()
        print('time for coordinate loading: ',start-mid)
            
        sub_medians, sub_medians_r_vir, inside_galaxy, inside_halo, star_formation_dist, profiles_hmr, profiles_r_vir, profiles_situ_hmr,\
        profiles_situ_r_vir =\
        distances(parent_indices_data, final_offsets, all_gas_pos, all_star_pos, sub_pos_at_target_snap, subhaloFlag, sub_ids, shmr, r_vir,\
                  boxSize, star_formation_snaps, target_snap, shmr_cut, r_vir_cut, accretion_channels, situ_cat, return_profiles, num_hmr,\
                  num_r_vir, numBins)
        
        #convert to cumulative profiles if requested
        if cumulative and return_profiles:
            profiles_hmr = np.cumsum(profiles_hmr, axis = 2)
            profiles_r_vir = np.cumsum(profiles_r_vir, axis = 2) ### change to axis = 1 for reduced memory usage
            if stype == 'insitu':
                pass
                profiles_situ_hmr = np.cumsum(profiles_situ_hmr, axis = 2)
                profiles_situ_r_vir = np.cumsum(profiles_situ_r_vir, axis = 2)
        
    else:
        all_dm_pos = il.snapshot.loadSubset(basePath, target_snap, 'dm', fields = ['Coordinates'])
        
        start = time.time()
        print('time for coordinate loading: ',start-mid)
        
        sub_medians, sub_medians_r_vir, inside_galaxy, inside_halo, profiles_hmr, profiles_r_vir =\
        distances_dm(parent_indices_data, final_offsets, all_dm_pos, sub_pos_at_target_snap, subhaloFlag, sub_ids, shmr, r_vir, boxSize,\
                     target_snap, shmr_cut, r_vir_cut, return_profiles, num_hmr, num_r_vir, numBins)
        
        star_formation_dist = None
    
        #convert to cumulative profiles if requested
        if cumulative and return_profiles:
            profiles_hmr = np.cumsum(profiles_hmr, axis = 2)
            profiles_r_vir = np.cumsum(profiles_r_vir, axis = 2)

        profiles_situ_hmr = np.zeros(1)
        profiles_situ_r_vir = np.zeros(1)

    end = time.time()
    print('actual time for profiles: ',end-start)
    return sub_medians, sub_medians_r_vir, subhaloFlag, inside_galaxy, inside_halo, star_formation_dist, sub_ids_dwarfs, sub_ids_mw,\
sub_ids_groups, sub_ids_giants, profiles_hmr, profiles_r_vir, profiles_situ_hmr, profiles_situ_r_vir

#---- settings----#
run = int(sys.argv[1])
stype = str(sys.argv[2])
target_snap = int(sys.argv[3])
basePath='/virgotng/universe/IllustrisTNG/TNG50-' + str(run) + '/output'
start_snap = 99
use_sfr_gas_hmr = True
return_profiles = True
cumulative = True

shmr_cut = 2
r_vir_cut = 1

numBins = 101 #201 for radial profile evolution, 101 for radial profiles
num_r_vir = 15 #1.5 for radial profile evolution, 15 for radial profiles
num_hmr = int(num_r_vir * 40 / 3) #200shmr or 15 r_vir are good numbers
dist_bins_hmr = np.linspace(0,num_hmr,numBins)
dist_bins_r_vir = np.linspace(0,num_r_vir,numBins)
start = time.time()

assert isdir('/vera/ptmp/gc/olwitt/' + stype + '/' +basePath[32:39] + '/lagrangian_regions')

sub_medians, sub_medians_r_vir, subhaloFlag, inside_galaxy, inside_halo, star_formation_dist, dwarf_inds, mw_inds, group_inds,\
giant_inds, cum_rad_prof_hmr, cum_rad_prof_r_vir, profiles_situ_hmr, profiles_situ_r_vir =\
lagrangian_region(basePath, stype, start_snap, target_snap, shmr_cut, r_vir_cut, use_sfr_gas_hmr, return_profiles, num_hmr, num_r_vir,\
                  numBins, cumulative)

user = '/vera/ptmp/gc/olwitt'


filename = user + '/' + stype + '/' + basePath[32:39] + '/lagrangian_regions/lagrangian_regions_'

if return_profiles:
    filename = filename + f'w_profiles_{target_snap}_cumulative.hdf5'
else:
    filename = filename + f'{target_snap}.hdf5'


f = h5py.File(filename,'w')

f.create_dataset('lagrangian_regions_shmr',data = sub_medians)
f.create_dataset('lagrangian_regions_r_vir',data = sub_medians_r_vir)

if return_profiles:
    if cumulative:
        f.create_dataset('cumulative_radial_profiles_hmr', data = cum_rad_prof_hmr)
        f.create_dataset('cumulative_radial_profiles_r_vir', data = cum_rad_prof_r_vir)
        if stype == 'insitu':
            f.create_dataset('cumulative_radial_profiles_situ_hmr', data = profiles_situ_hmr)
            f.create_dataset('cumulative_radial_profiles_situ_r_vir', data = profiles_situ_r_vir)
    else:
        f.create_dataset('radial_profiles_hmr', data = cum_rad_prof_hmr)
        f.create_dataset('radial_profiles_r_vir', data = cum_rad_prof_r_vir)
        if stype == 'insitu':
            f.create_dataset('radial_profiles_situ_hmr', data = profiles_situ_hmr)
            f.create_dataset('radial_profiles_situ_r_vir', data = profiles_situ_r_vir)

f.create_dataset('subhaloFlag', data = subhaloFlag)
f.create_dataset('tracers_inside_galaxy', data = inside_galaxy)
f.create_dataset('tracers_inside_halo', data = inside_halo)

if stype.lower() not in ['dm', 'darkmatter', 'dark_matter', 'dark matter']:
    f.create_dataset('distance_at_star_formation', data = star_formation_dist)

g = f.create_group('mass_bin_sub_ids')
g.create_dataset('dwarfs', data = dwarf_inds)
g.create_dataset('mws', data = mw_inds)
g.create_dataset('groups', data = group_inds)
g.create_dataset('giants', data = giant_inds)

f.close()

