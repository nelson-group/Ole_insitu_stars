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
from scipy.spatial import ConvexHull as ch

sys.path.append('/vera/u/olwitt/illustris_python/illustris_python')
from loadMPBs import loadMPBs

@jit(nopython = True, parallel = True)
def isSubGalaxy(sub_ids, final_offsets):
    """ Checks the number of tracers in each galaxy and marks the ones with 0 tracers."""
    
    noGalaxy = np.ones(sub_ids.shape[0], dtype = np.ubyte)
    for i in nb.prange(sub_ids.shape[0]):
        sub_id = sub_ids[i]
        if final_offsets[sub_id + 1] - final_offsets[sub_id] < 4:
            continue
        noGalaxy[i] = 0
    return noGalaxy

def lagrangian_region_convex_hull(coords, final_offsets, sub_ids, subhaloFlag):
    num_subs = sub_ids.shape[0]
    volumes = np.zeros(num_subs, dtype = np.float32)
    radii = np.zeros(num_subs, dtype = np.float32)
    
    for i in range(num_subs):
        if subhaloFlag[i] == 0:
            continue
        indices_of_sub = np.arange(final_offsets[sub_ids[i]],final_offsets[sub_ids[i]+1])
        coords_in_sub = coords[indices_of_sub]
        
        #check whether all tracers are in the same star -> no convex hull can be computed
        unique_coords = np.unique(coords_in_sub, axis = 0)
        
        if unique_coords.shape[0] < 4:
            continue
        
        hull = ch(coords_in_sub)
        volumes[i] = hull.volume
        radii[i] = (3*hull.volume / 4 / np.pi)**(1/3)#??
        #also possible:
#         radii[i] = 3 * hull.volume / hull.area
    
    return volumes, radii


@jit(nopython = True, parallel = True)
def distances(final_offsets, part_pos, sub_pos_at_target_snap, subhaloFlag, sub_ids, boxSize, target_snap, accretion_channels,\
              situ_cat, stype):
    """
    For each central galaxy with >3 tracers, the distance of every tracer to the (MP) subhalo center is computed. Furthermore, it computes the
    median and maximum distance of all tracers of a galaxy.
    """
    
    num_subs = sub_ids.shape[0]
    
    sub_medians = np.zeros((num_subs,3,3), dtype = np.float32)
    sub_max = sub_medians.copy()
        
    for i in nb.prange(num_subs):
        
        #skip unsuitable subhalos 
        if subhaloFlag[i] == 0:
            continue
            
        sub_id = sub_ids[i]
        indices_of_sub = np.arange(final_offsets[sub_id],final_offsets[sub_id+1])
        
        
#         parent_indices_of_sub = parent_indices_data[indices_of_sub,:]

        part_pos_of_sub = part_pos[indices_of_sub,:]
        
        subhalo_position = sub_pos_at_target_snap[sub_id,:] 

        rad_dist = funcs.dist_vector_nb(subhalo_position,part_pos_of_sub,boxSize)
        
        if stype == 'insitu':
        
            #igm_mask: all tracers directly from the igm, i.e. from fresh accretion ('0') or nep wind recycling ('1')
            #satellite_mask: all tracers that entered via mergers ('2')
            ac_ch_at_cut = accretion_channels[indices_of_sub]
            igm_mask = np.where(funcs.isin(ac_ch_at_cut,np.array([0,1])))[0]
            satellite_mask = np.where(ac_ch_at_cut == 2)[0]

            #Lagrangian region computations:
            for j in range(3):
                if j == 0:
                    #all tracers
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

                sub_medians[i,j,0] = np.median(rad_dist[subset])
                sub_max[i,j,0] = np.max(rad_dist[subset])

                if subset_igm_mask.size > 0:
                    sub_medians[i,j,1] = np.median(rad_dist[subset_igm_mask])
                    sub_max[i,j,1] = np.max(rad_dist[subset_igm_mask])

                if subset_satellite_mask.size > 0:
                    sub_medians[i,j,2] = np.median(rad_dist[subset_satellite_mask])
                    sub_max[i,j,2] = np.max(rad_dist[subset_satellite_mask])
        else:
            sub_medians[i,0,0] = np.median(rad_dist)
            sub_max[i,0,0] = np.max(rad_dist)
        
    return sub_medians, sub_max

def lagrangian_region_cat(basePath, stype, start_snap, target_snap):
    start_loading = time.time()
    header = il.groupcat.loadHeader(basePath,target_snap)
    redshift = header['Redshift']
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
    all_ids = np.arange(group_masses.shape[0])

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
    
    #load data from files ---------------------------------------------------------------------------------
    sub_positions = h5py.File('files/'+basePath[32:39]+'/SubhaloPos_new_extrapolated.hdf5','r') 
    is_extrapolated = sub_positions['is_extrapolated'][:]
    
    #load subhalo positions (99 instead of start_snap as they were computed for start_snap = 99)
    sub_pos_at_target_snap = sub_positions['SubhaloPos'][:,99-target_snap,:]
    sub_positions.close()
    
    if stype.lower() in ['insitu', 'exsitu', 'in', 'ex', 'in-situ', 'ex-situ']:
        if stype.lower() in ['insitu', 'in', 'in-situ']:
            stype = 'insitu'
        else:
            stype = 'exsitu'
        
        
        file = '/vera/ptmp/gc/olwitt/' + stype + '/' + basePath[32:39] + f'/parent_indices_{target_snap}.hdf5'
        f = h5py.File(file,'r')

        parent_indices = f[f'snap_{target_snap}/parent_indices'][:,:]
        parent_indices_data = parent_indices[:,:].astype(int)
        num_tracers = parent_indices.shape[0]
    
    #offsets -----------------------------------------------------------------------------------------------
        #here, it's okay that the offsets at the target snapshot are used as they are identical at every snapshot
        if f.__contains__(f'snap_{target_snap}/numTracersInParents'):
            numTracersInParents = f[f'snap_{target_snap}/numTracersInParents'][:]
        else:
            numTracersInParents = f[f'snap_{target_snap}/tracers_in_parents_offset'][:]
        f.close()
        
        if stype == 'insitu':
            insituStarsInSubOffset = tF.insituStarsInSubOffset(basePath,start_snap)
            
            file = f'/vera/ptmp/gc/olwitt/auxCats/TNG50-{run}/insitu_or_medsitu_{start_snap}.hdf5'
            f = h5py.File(file,'r')
            #0: insitu, 1: medsitu
            situ_cat = f['stellar_assembly'][:]
            f.close()
        else:
            insituStarsInSubOffset = tF.exsituStarsInSubOffset(basePath,start_snap)
            situ_cat = None
            
        final_offsets = tF.tracersInSubhalo(insituStarsInSubOffset,numTracersInParents).astype(int)
        final_offsets = np.insert(final_offsets,0,0)
        
        del insituStarsInSubOffset, numTracersInParents
        
        #load accretion channels for tracers
        file = '/vera/ptmp/gc/olwitt/auxCats/' + basePath[32:39] + f'/tracer_accretion_channels_{start_snap}.hdf5'
        f = h5py.File(file,'r')
        accretion_channels = f['tracer_accretion_channels'][:]
        f.close()
        
        all_gas_pos = il.snapshot.loadSubset(basePath,target_snap, 'gas', fields = ['Coordinates'])
#         all_star_pos = il.snapshot.loadSubset(basePath,target_snap, 'stars', fields = ['Coordinates'])
        
#         if isinstance(all_star_pos, dict):
#             all_star_pos = None
        
        part_pos = all_gas_pos[parent_indices_data[:,0]]
        
    elif stype.lower() in ['dm', 'darkmatter', 'dark_matter', 'dark matter']:
        stype = 'dm'
        
        file = '/vera/ptmp/gc/olwitt/' + stype + '/' + basePath[32:39] + f'/dm_indices_{target_snap}.hdf5'
        f = h5py.File(file,'r')

        parent_indices = f[f'dm_indices'][:]
        parent_indices_data = parent_indices.astype(int)
        num_tracers = parent_indices.shape[0]
        
        final_offsets = f['dmInSubOffset'][:]
        f.close()
        
        all_dm_pos = il.snapshot.loadSubset(basePath, target_snap, 'dm', fields = ['Coordinates'])
        
        accretion_channels = None
        situ_cat = None
        part_pos = all_dm_pos[parent_indices_data]
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
    
    #### ^ only exclude galxies without tracers and no extrapolated position history ^ ####

    end_loading = time.time()
    print('time for coordinate loading: ', end_loading - start_loading)

    sub_medians, sub_max =\
    distances(final_offsets, part_pos, sub_pos_at_target_snap, subhaloFlag, sub_ids, boxSize, target_snap, accretion_channels,\
          situ_cat, stype)
    
    if stype != 'insitu':
        sub_medians = sub_medians[:,0,0]
        sub_max = sub_max[:,0,0]
    
    end_distances = time.time()
    print('time for distances: ',end_distances - end_loading)
    
    volumes, radii = lagrangian_region_convex_hull(part_pos, final_offsets, sub_ids, subhaloFlag)
        
    end_hull = time.time()
    print('time for volumes: ',end_hull - end_distances)
    return subhaloFlag, sub_ids_dwarfs, sub_ids_mw, sub_ids_groups, sub_ids_giants, sub_medians, sub_max, volumes, radii


#---- settings----#
run = int(sys.argv[1])
stype = str(sys.argv[2])
#only consider snapshot 0
target_snap = 0
basePath='/virgotng/universe/IllustrisTNG/TNG50-' + str(run) + '/output'
start_snap = 99

start = time.time()

assert isdir('/vera/ptmp/gc/olwitt/auxCats/' +basePath[32:39] )

subhaloFlag, dwarf_inds, mw_inds, group_inds, giant_inds, sub_medians, sub_max, volumes, radii = lagrangian_region_cat(basePath, stype, start_snap, target_snap)

user = '/vera/ptmp/gc/olwitt'


filename = user + '/auxCats/' + basePath[32:39] + f'/baryonic_lagrangian_regions{stype}.hdf5'


f = h5py.File(filename,'w')
f.create_dataset('subhalo_median_distances', data = sub_medians)
f.create_dataset('subhalo_maximum_distances', data = sub_max)
f.create_dataset('subhaloFlag', data = subhaloFlag)

g1 = f.create_group('convex_hull')
g1.create_dataset('volumes', data = volumes)
g1.create_dataset('radii', data = radii)

g2 = f.create_group('mass_bin_sub_ids')
g2.create_dataset('dwarfs', data = dwarf_inds)
g2.create_dataset('mws', data = mw_inds)
g2.create_dataset('groups', data = group_inds)
g2.create_dataset('giants', data = giant_inds)

f.close()

