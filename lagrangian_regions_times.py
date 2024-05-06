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

def find_star_formation_snap(basePath, stype, start_snap, num_tracers):
    """Computes snapshot of star formation for every tracer based on the parent index tables."""
    
    snaps = np.arange(start_snap, -1,-1)
    star_formation_snap = np.full(num_tracers, -1, dtype = np.byte)
    
    for i, snap in enumerate(snaps):
        file = '/vera/ptmp/gc/olwitt/' + stype + '/' + basePath[32:39] + f'/parent_indices_{snap}.hdf5'
        f = h5py.File(file,'r')
        #only load information of state of tracer parent (gas/star)
        new_parent_indices = f[f'snap_{snap}/parent_indices'][:,1]
        f.close()
        
        if i > 0:
            #save last snapshot of star formation, i.e. the highest snapshot at which the gas parent turns into a star
            new_star = np.where(np.logical_and(np.logical_and(new_parent_indices == 0, old_parent_indices == 1), formation_snap == -1))
            formation_snap[new_star] = snap + 1
        
        old_parent_indices = new_parent_indices.copy()
    
    return star_formation_snap

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

@jit(nopython = True, parallel = True)
def distances(parent_indices_data, location_at_cut, isInMP_at_cut, final_offsets, all_gas_pos, all_star_pos,\
                      sub_pos_at_target_snap, subhaloFlag, sub_ids, shmr, r_vir, boxSize, star_formation_snaps, target_snap):
    """ For each galaxy with >1 tracers, the distance of every tracer to the (MP) subhalo center is computed. Furthermore, it checks whether the tracers is located within the halo or even the galaxy."""
    
    sub_medians = np.full((sub_ids.shape[0],3),np.nan)
    sub_medians_r_vir = np.full((sub_ids.shape[0],3),np.nan)
    
    inside_2shmr = np.zeros(parent_indices_data.shape[0], dtype = np.ubyte)
    inside_r_vir = np.zeros(parent_indices_data.shape[0], dtype = np.ubyte)
    
    num_new_stars = np.where(star_formation_snaps == target_snap)[0].shape[0]
    dist_at_star_form = np.zeros(num_new_stars)
    
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

    #the following assertion is not fulfilled as we are only considering centrals. star_formation_snaps includes tracers of every subhalo
#     assert star_form_offsets[-1] == num_new_stars
        
    for i in nb.prange(sub_ids.shape[0]):
        
        #skip unsuitable subhalos 
        if subhaloFlag[i] == 0:
            continue
            
        sub_id = sub_ids[i]
        indices_of_sub = np.arange(final_offsets[sub_id],final_offsets[sub_id+1])
        location_of_sub_at_cut = location_at_cut[indices_of_sub]
        isInMP_of_sub_at_cut = isInMP_at_cut[indices_of_sub]
        
        parent_indices_of_sub = parent_indices_data[indices_of_sub,:]

        particle_pos = np.zeros((indices_of_sub.shape[0],3))
        gas_mask = np.where(parent_indices_of_sub[:,1] == 0)[0]
        star_mask = np.where(parent_indices_of_sub[:,1] == 1)[0]
        
        gas_parent_indices = parent_indices_of_sub[gas_mask,0]
        particle_pos[gas_mask,:] = all_gas_pos[gas_parent_indices,:]

        star_parent_indices = parent_indices_of_sub[star_mask,0]
        particle_pos[star_mask,:] = all_star_pos[star_parent_indices,:]
        
        #prior: sub_id instead of index!!!
        subhalo_position = sub_pos_at_target_snap[sub_id,:] 

        rad_dist = funcs.dist_vector_nb(subhalo_position,particle_pos,boxSize)
        
        #radius crossings:
        
        in_gal = np.where(rad_dist < 2 * shmr[i])[0]
        inside_2shmr[indices_of_sub[in_gal]] = 1
        in_halo = np.where(rad_dist < r_vir[i])[0]
        inside_r_vir[indices_of_sub[in_halo]] = 1
        
        #radius at star formation (normalized by shmr):
        
        new_stars_in_sub = np.where(star_formation_snaps[indices_of_sub] == target_snap)[0]
        num_new_stars_in_sub = new_stars_in_sub.shape[0]
        dist_at_star_form[star_form_offsets[i]:star_form_offsets[i+1]] = rad_dist[new_stars_in_sub] / shmr[i]
        
        #Lagrangian region computations:
        
        igm_mask = np.where(location_of_sub_at_cut == -1)[0]
        satellite_mask = np.where((location_of_sub_at_cut != -1) & (np.logical_not(isInMP_of_sub_at_cut)))[0]
                
        if rad_dist.size > 0:
            sub_medians[i,0] = np.median(rad_dist) / shmr[i]
            
        if igm_mask.size > 0:
            sub_medians[i,1] = np.median(rad_dist[igm_mask]) / shmr[i]

        if satellite_mask.size > 0:
            sub_medians[i,2] = np.median(rad_dist[satellite_mask]) / shmr[i]
            
        if rad_dist.size > 0:
            sub_medians_r_vir[i,0] = np.median(rad_dist) / r_vir[i]
            
        if igm_mask.size > 0:
            sub_medians_r_vir[i,1] = np.median(rad_dist[igm_mask]) / r_vir[i]

        if satellite_mask.size > 0:
            sub_medians_r_vir[i,2] = np.median(rad_dist[satellite_mask]) / r_vir[i]
            
        
    return sub_medians, sub_medians_r_vir, inside_2shmr, inside_r_vir, dist_at_star_form

def lagrangian_region(basePath, stype, start_snap, target_snap, cut_snap):
    start_loading = time.time()
    header = il.groupcat.loadHeader(basePath,target_snap)
    redshift = header['Redshift']
    h_const = header['HubbleParam']
    boxSize = header['BoxSize']
    num_subs = il.groupcat.loadHeader(basePath,start_snap)['Nsubgroups_Total']

    #introduce mass bins (just for analysis, not for computation):
    groups = il.groupcat.loadHalos(basePath, start_snap, fields = ['Group_M_Crit200','GroupFirstSub'])
    group_masses = groups['Group_M_Crit200']*1e10/h_const

    #differentiate between halos of dwarf / milky way / group size
    dwarf_ids = np.where(np.logical_and(group_masses > 10**(10.8), group_masses < 10**(11.2)))[0]
    mw_ids = np.where(np.logical_and(group_masses > 10**(11.8), group_masses < 10**(12.2)))[0]
    group_ids = np.where(np.logical_and(group_masses > 10**(12.6), group_masses < 10**(13.4)))[0]
    giant_ids = np.where(group_masses > 10**(13.4))[0]
    all_ids = np.arange(group_masses.shape[0])

    #find ids of associated centrals
    sub_ids_dwarfs = groups['GroupFirstSub'][dwarf_ids]
    sub_ids_mw = groups['GroupFirstSub'][mw_ids]
    sub_ids_groups = groups['GroupFirstSub'][group_ids]
    sub_ids_giants = groups['GroupFirstSub'][giant_ids]
    all_central_ids = groups['GroupFirstSub'][:]
    
    dwarf_inds = tF.getIndices(sub_ids_dwarfs, all_central_ids)
    mw_inds = tF.getIndices(sub_ids_mw, all_central_ids)
    group_inds = tF.getIndices(sub_ids_groups, all_central_ids)
    giant_inds = tF.getIndices(sub_ids_giants, all_central_ids)
    
    del groups, group_masses, dwarf_ids, mw_ids, group_ids, giant_ids, sub_ids_dwarfs, sub_ids_mw, sub_ids_groups, sub_ids_giants

    sub_ids = all_central_ids
    tree_ids = np.arange(num_subs)
#     sub_ids = sub_ids_mw if halo_type == 'mw' else sub_ids_dwarves if halo_type == 'dwarves' else sub_ids_groups if halo_type == 'groups' else all_central_ids

#     ids = mw_ids if halo_type == 'mw' else dwarf_ids if halo_type == 'dwarves' else group_ids if halo_type == 'groups' else all_ids

    #Filter out halos without any subhalo
    subhaloFlag = np.ones(sub_ids.shape[0], dtype = np.ubyte)
    subhaloFlag[np.where(sub_ids == -1)] = 0
    
    #load MPB trees
    trees = loadMPBs(basePath, start_snap, ids = tree_ids, fields = ['SubfindID'])
    
    #load data from files ---------------------------------------------------------------------------------
    file = '/vera/ptmp/gc/olwitt/' + stype + '/' + basePath[32:39] + f'/parent_indices_{target_snap}.hdf5'
    f = h5py.File(file,'r')
    
    sub_positions = h5py.File('files/'+basePath[32:39]+'/SubhaloPos_new_extrapolated.hdf5','r') 
    is_extrapolated = sub_positions['is_extrapolated'][:]
    
    #load subhalo positions (99 instead of start_snap as they were computed for start_snap = 99)
    sub_pos_at_target_snap = sub_positions['SubhaloPos'][:,99-target_snap,:]
    sub_positions.close()
    
    parent_indices = f[f'snap_{target_snap}/parent_indices'][:,:]
    parent_indices_data = parent_indices[:,:].astype(int)
    num_tracers = parent_indices.shape[0]
    
    loc_file = h5py.File(f'/vera/ptmp/gc/olwitt/' + stype + '/' + basePath[32:39] + '/subhalo_index_table.hdf5','r')
    location = loc_file[f'snap_{cut_snap}/location'][:]
    
    location_type = loc_file[f'snap_{cut_snap}/location_type'][:]
    isInMP = np.zeros(location_type.shape[0], dtype = np.ubyte)
    isInMP[np.isin(location_type,np.array([1,2]))] = 1
    del location_type
    loc_file.close()
    
    #offsets -----------------------------------------------------------------------------------------------
    if f.__contains__(f'snap_{target_snap}/numTracersInParents'):
        numTracersInParents = f[f'snap_{target_snap}/numTracersInParents'][:]
    else:
        numTracersInParents = f[f'snap_{target_snap}/tracers_in_parents_offset'][:]
    f.close()
        
    if stype == 'insitu':
        insituStarsInSubOffset = tF.insituStarsInSubOffset(basePath,start_snap)
    elif stype == 'exsitu':
        insituStarsInSubOffset = tF.exsituStarsInSubOffset(basePath,start_snap)
    else:
        raise Exception('Invalid star type!')
    final_offsets = tF.tracersInSubhalo(insituStarsInSubOffset,numTracersInParents).astype(int)
    final_offsets = np.insert(final_offsets,0,0)
    # ^ offsets for the parent index table, that's why at snapshot 99
    
    del insituStarsInSubOffset, numTracersInParents
    
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
    
    #<until here, subhaloFlag is identical for every snapshot>
    
    #now aquire the correct virial radii (consider only those galaxies that are still centrals):
    #-2 bc. GroupFirstSub could contain -1's
    shmr = np.zeros(sub_ids.shape[0], dtype = float)
    target_sub_ids = np.full(sub_ids.shape[0], -2, dtype = int)
    r_vir = np.zeros(sub_ids.shape[0], dtype = float)
    
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
        print(f'No groups at snapshot {target_snap}! -> return = 10*[-1]')
        return 10*(np.array([-1]),)
        
    central_sub_ids_at_target_snap, GFS_inds, TSID_inds = np.intersect1d(groupFirstSub, target_sub_ids, return_indices = True)
    r_vir_cat = il.groupcat.loadHalos(basePath, target_snap, fields = ['Group_R_Crit200'])[GFS_inds]
    r_vir[TSID_inds] = r_vir_cat
    shmr_cat = il.groupcat.loadSubhalos(basePath, target_snap, fields = ['SubhaloHalfmassRadType'])[central_sub_ids_at_target_snap,4]
    shmr[TSID_inds] = shmr_cat

    #only keep subhalos that are still centrals
    mask = np.full(sub_ids.shape[0], True)
    mask[TSID_inds] = False
    subhaloFlag[mask] = 0 
    
    del TSID_inds, GFS_inds, central_sub_ids_at_target_snap, r_vir_cat, shmr_cat
    
    
    ######## this necessary?? #########
#     zero_shmr = np.where(shmr <= 0.0001)[0]
#     zero_r_vir = np.where(r_vir <= 0.1)[0]
    zero_shmr = np.where(shmr == 0)[0]
    zero_r_vir = np.where(r_vir == 0)[0]
    
    subhaloFlag[zero_shmr] = 0
    subhaloFlag[zero_r_vir] = 0
#     #exclude all galaxies with shmr smaller than 0.0001ckpc, i.e. essentially zero ckpc
    
#     #all galaxies without extrapolated sub_pos history or only 1 tracer or having a vanishing shmr: -1
#     extrapolated_sub_ids[zero_r_norm] = -1
#     #all galaxies without extrapolated sub_pos history or only 1 tracer or having a vanishing r_vir or being a satellite: -1
#     extrapolated_central_sub_ids[zero_r_vir] = -1
    
    #get star formation snapshot for all tracers
    
    f = h5py.File('/vera/ptmp/gc/olwitt/' + stype + '/' + basePath[32:39] + '/star_formation_snapshots.hdf5','r')
    star_formation_snaps = f['star_formation_snapshot'][:]
    f.close()
    
    #load particle positions only right before computation begins
    all_gas_pos = il.snapshot.loadSubset(basePath,target_snap, 'gas', fields = ['Coordinates'])
    all_star_pos = il.snapshot.loadSubset(basePath,target_snap, 'stars', fields = ['Coordinates'])
    
    #check memory usage
#     print(psutil.virtual_memory().percent,' % of RAM used')
    
    if isinstance(all_star_pos, dict):
        all_star_pos = np.zeros((1,3))
    
    start = time.time()
    print('time for loading and shit: ',start-start_loading)
    
    
    sub_medians, sub_medians_r_vir, inside_galaxy, inside_halo, star_formation_dist =\
    distances(parent_indices_data, location, isInMP, final_offsets, all_gas_pos, all_star_pos, sub_pos_at_target_snap, subhaloFlag,\
              sub_ids, shmr, r_vir, boxSize, star_formation_snaps, target_snap)
    
    
    end = time.time()
    print('actual time for profiles: ',end-start)
    return sub_medians, sub_medians_r_vir, subhaloFlag, inside_galaxy, inside_halo, star_formation_dist, dwarf_inds, mw_inds, group_inds,\
giant_inds

#---- settings----#
run = int(sys.argv[1])
stype = str(sys.argv[2])
target_snap = int(sys.argv[3])
cut_snap = int(sys.argv[4])
basePath='/virgotng/universe/IllustrisTNG/TNG50-' + str(run) + '/output'
start_snap = 99
start = time.time()

assert isdir('/vera/ptmp/gc/olwitt/' + stype + '/'+basePath[32:39]+'/lagrangian_regions')

sub_medians, sub_medians_r_vir, subhaloFlag, inside_galaxy, inside_halo, star_formation_dist, dwarf_inds, mw_inds, group_inds,\
giant_inds = lagrangian_region(basePath, stype, start_snap, target_snap, cut_snap)

f = h5py.File('/vera/ptmp/gc/olwitt/' + stype + '/'+basePath[32:39]+\
              f'/lagrangian_regions/lagrangian_regions_cut{cut_snap}_{target_snap}.hdf5','w')

f.create_dataset('lagrangian_regions_shmr',data = sub_medians)
f.create_dataset('lagrangian_regions_r_vir',data = sub_medians_r_vir)

f.create_dataset('subhaloFlag', data = subhaloFlag)
f.create_dataset('tracers_inside_galaxy', data = inside_galaxy)
f.create_dataset('tracers_inside_halo', data = inside_halo)
f.create_dataset('distance_at_star_formation', data = star_formation_dist)

g = f.create_group('mass_bin_indices')
g.create_dataset('dwarf_indices', data = dwarf_inds)
g.create_dataset('mw_indices', data = mw_inds)
g.create_dataset('group_indices', data = group_inds)
g.create_dataset('giant_indices', data = giant_inds)

f.close()

