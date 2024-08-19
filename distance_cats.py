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

sys.path.append('/vera/u/olwitt/illustris_python/illustris_python')
from loadMPBs import loadMPBs

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
def distances(parent_indices_data, final_offsets, all_gas_pos, all_star_pos, sub_pos_at_target_snap, subhaloFlag, sub_ids, shmr, r_vir, boxSize, stype, target_snap, star_formation_snaps):
    """ For each galaxy with >1 tracers, the distance of every tracer to the (MP) subhalo center is computed.
      Furthermore, it checks whether the tracers is located within the halo or even the galaxy.
      
      Output:
      -1: not in galaxy from sample
      0: outside halo (r > R_200c)
      1: inside outer halo (0.5 R_200c < r < R_200c)
      2: inside inner halo (0.25 R_200c < r < 0.5 R_200c)
      3: inside very inner halo (0.1 R_200c < r < 0.25 R_200c)
      4: inside galaxy (r < 0.1 R_200c)
      +100: inside 2shmr (r < 2 R_{SF, 1/2})
      """
    if stype == 'insitu':
        num_new_stars = np.where(star_formation_snaps == target_snap)[0].shape[0]
        dist_at_star_form = np.full(num_new_stars, -1, dtype = np.float32)
        
        #compute offsets so the second loop can run in parallel
        
        star_form_offsets = np.zeros(sub_ids.shape[0], dtype = np.int64)
        for i in nb.prange(sub_ids.shape[0]):
            indices_of_sub = np.arange(final_offsets[sub_ids[i]],final_offsets[sub_ids[i]+1])
            new_stars_in_sub = np.where(star_formation_snaps[indices_of_sub] == target_snap)[0]
            num_new_stars_in_sub = new_stars_in_sub.shape[0]
            star_form_offsets[i] = num_new_stars_in_sub
            
        star_form_offsets = np.cumsum(star_form_offsets)
        #numba compatible insert function:
        star_form_offsets = funcs.insert(star_form_offsets, 0, 0)

    else:
        star_form_offsets = np.zeros(1, dtype = np.int64)
        dist_at_star_form = np.zeros(1, dtype = np.float32)


    inside_radius = np.full(parent_indices_data.shape[0], -1, dtype = np.byte)
        
    for i in nb.prange(sub_ids.shape[0]):
        
        #skip unsuitable subhalos 
        if subhaloFlag[i] == 0:
            continue
            
        sub_id = sub_ids[i]
        indices_of_sub = np.arange(final_offsets[sub_id],final_offsets[sub_id+1])
        
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
        
        not_in_halo = np.where(rad_dist >= r_vir[i])[0]
        inside_radius[indices_of_sub[not_in_halo]] = 0
        in_outer_halo = np.where(np.logical_and(rad_dist <  r_vir[i], rad_dist > 0.5*r_vir[i]))[0]
        inside_radius[indices_of_sub[in_outer_halo]] = 1
        in_inner_halo = np.where(np.logical_and(rad_dist < 0.5*r_vir[i], rad_dist > 0.25*r_vir[i]))[0]
        inside_radius[indices_of_sub[in_inner_halo]] = 2
        in_very_inner_halo = np.where(np.logical_and(rad_dist < 0.25*r_vir[i], rad_dist > 0.1*r_vir[i]))[0]
        inside_radius[indices_of_sub[in_very_inner_halo]] = 3
        in_gal = np.where(rad_dist < 0.1*r_vir[i])[0]
        inside_radius[indices_of_sub[in_gal]] = 4
        in_2shmr = np.where(rad_dist < 2 * shmr[i])[0]
        inside_radius[indices_of_sub[in_2shmr]] += 100

        #radius at star formation (normalized by shmr):
        if stype == 'insitu':
            new_stars_in_sub = np.where(star_formation_snaps[indices_of_sub] == target_snap)[0]
            num_new_stars_in_sub = new_stars_in_sub.shape[0]
            dist_at_star_form[star_form_offsets[i]:star_form_offsets[i+1]] = rad_dist[new_stars_in_sub] / shmr[i]
        
    return inside_radius, dist_at_star_form

@jit(nopython = True, parallel = True)
def distances_dm(parent_indices_data, final_offsets, all_dm_pos, sub_pos_at_target_snap, subhaloFlag, sub_ids, shmr, r_vir, boxSize):
    """ For each galaxy with >1 tracers, the distance of every tracer to the (MP) subhalo center is computed.
      Furthermore, it checks whether the tracers is located within the halo or even the galaxy.
      
      Output:
      -1: not in galaxy from sample
      0: outside halo (r > R_200c)
      1: inside outer halo (0.5 R_200c < r < R_200c)
      2: inside inner halo (0.25 R_200c < r < 0.5 R_200c)
      3: inside very inner halo (0.1 R_200c < r < 0.25 R_200c)
      4: inside galaxy (r < 0.1 R_200c)
      +100: inside 2shmr (r < 2 R_{SF, 1/2})
      """
    

    inside_radius = np.full(parent_indices_data.shape[0], -1, dtype = np.byte)
    
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

        
        not_in_halo = np.where(rad_dist >= r_vir[i])[0]
        inside_radius[indices_of_sub[not_in_halo]] = 0
        in_outer_halo = np.where(np.logical_and(rad_dist <  r_vir[i], rad_dist > 0.5*r_vir[i]))[0]
        inside_radius[indices_of_sub[in_outer_halo]] = 1
        in_inner_halo = np.where(np.logical_and(rad_dist < 0.5*r_vir[i], rad_dist > 0.25*r_vir[i]))[0]
        inside_radius[indices_of_sub[in_inner_halo]] = 2
        in_very_inner_halo = np.where(np.logical_and(rad_dist < 0.25*r_vir[i], rad_dist > 0.1*r_vir[i]))[0]
        inside_radius[indices_of_sub[in_very_inner_halo]] = 3
        in_gal = np.where(rad_dist < 0.1*r_vir[i])[0]
        inside_radius[indices_of_sub[in_gal]] = 4
        in_2shmr = np.where(rad_dist < 2 * shmr[i])[0]
        inside_radius[indices_of_sub[in_2shmr]] += 100
    
    return inside_radius

def distance_cats(basePath, stype, start_snap, target_snap, use_sfr_gas_hmr):
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
        
    elif stype.lower() in ['dm', 'darkmatter', 'dark_matter', 'dark matter']:
        stype = 'dm'
        
        file = '/vera/ptmp/gc/olwitt/' + stype + '/' + basePath[32:39] + f'/dm_indices_{target_snap}.hdf5'
        f = h5py.File(file,'r')

        parent_indices = f[f'dm_indices'][:]
        parent_indices_data = parent_indices.astype(int)
        
        final_offsets = f['dmInSubOffset'][:]
        f.close()
        
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
        print(f'No groups at snapshot {target_snap}! -> return = 3*[-1]')
        return 3*(np.array([-1]),)
        
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
        # load starformation snapshots to compute star formation distances
        if stype == 'insitu':
            f = h5py.File('/vera/ptmp/gc/olwitt/' + stype + '/' + basePath[32:39] + '/star_formation_snapshots.hdf5','r')
            star_formation_snaps = f['star_formation_snapshot'][:]
            f.close()
        else:
            star_formation_snaps = None


        # load particle positions only right before computation begins
        all_gas_pos = il.snapshot.loadSubset(basePath,target_snap, 'gas', fields = ['Coordinates'])
        all_star_pos = il.snapshot.loadSubset(basePath,target_snap, 'stars', fields = ['Coordinates'])
    
        if isinstance(all_star_pos, dict):
            all_star_pos = np.zeros((1,3))
            
        start = time.time()
        print('time for coordinate loading: ',start-mid)
            
        inside_radius, star_formation_distances = distances(parent_indices_data, final_offsets, all_gas_pos, all_star_pos, sub_pos_at_target_snap, subhaloFlag, sub_ids, shmr, r_vir, boxSize,\
                                   stype, target_snap, star_formation_snaps)
        
    else:
        all_dm_pos = il.snapshot.loadSubset(basePath, target_snap, 'dm', fields = ['Coordinates'])
        
        start = time.time()
        print('time for coordinate loading: ',start-mid)
        
        inside_radius = distances_dm(parent_indices_data, final_offsets, all_dm_pos, sub_pos_at_target_snap, subhaloFlag, sub_ids, shmr, r_vir, boxSize)
        star_formation_distances = None

    end = time.time()
    print('actual time for distance sorting: ',end-start)
    return inside_radius, subhaloFlag, final_offsets, star_formation_distances

#---- settings----#
run = int(sys.argv[1])
stype = str(sys.argv[2])
target_snap = int(sys.argv[3])
basePath='/virgotng/universe/IllustrisTNG/TNG50-' + str(run) + '/output'
start_snap = 99
use_sfr_gas_hmr = True

start = time.time()

assert isdir('/vera/ptmp/gc/olwitt/' + stype + '/' + basePath[32:39] + '/distance_cats/')

inside_radius, subhaloFlag, final_offsets, star_formation_distances = distance_cats(basePath, stype, start_snap, target_snap, use_sfr_gas_hmr)

user = '/vera/ptmp/gc/olwitt'


filename = '/vera/ptmp/gc/olwitt/' + stype + '/' + basePath[32:39] + f'/distance_cats/distance_cats_{target_snap}.hdf5'

f = h5py.File(filename,'w')

f.create_dataset('tracers_inside_radius',data = inside_radius)
f.create_dataset('subhalo_offsets',data = final_offsets)
f.create_dataset('subhaloFlag', data = subhaloFlag)
f.create_dataset('aux_star_formation_distances', data = star_formation_distances)

f.close()

