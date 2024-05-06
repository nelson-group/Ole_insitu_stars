import time
import illustris_python as il
import numpy as np
import h5py
import numba as nb
from numba import jit, njit
import tracerFuncs as tF
import funcs
from os.path import isfile
import sys

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

@jit(nopython = True)
def mass_bin_profiles(parent_indices_data, final_offsets, all_gas_pos, all_star_pos, onlyGas, start_snap,\
                      target_snap, sub_pos_at_target_snap, extrapolated_sub_ids, num_bins, sub_ids, groupRcrit200,\
                      num_grc200):
    
    profiles = np.zeros((extrapolated_sub_ids.shape[0], num_bins))
    sub_means = np.zeros(extrapolated_sub_ids.shape[0])
    
    for index in nb.prange(extrapolated_sub_ids.shape[0]):
        sub_id = sub_ids[extrapolated_sub_ids[index]]
        parent_indices_of_sub =\
        parent_indices_data[final_offsets[sub_id]:final_offsets[sub_id+1],:]

        gas_parent_indices = parent_indices_of_sub[np.where(parent_indices_of_sub[:,1] == 0)[0],0]
        gas_pos = all_gas_pos[gas_parent_indices]

        if not onlyGas:
            star_parent_indices = parent_indices_of_sub[np.where(parent_indices_of_sub[:,1] == 1)[0],0]
            star_pos = all_star_pos[star_parent_indices]

        subhalo_position = sub_pos_at_target_snap[sub_id,start_snap-target_snap,:]

        rad_dist = np.zeros(gas_pos.shape[0])
        if not onlyGas:
            rad_dist = np.zeros(gas_pos.shape[0] + star_pos.shape[0])
        for j in nb.prange(gas_pos.shape[0]):
            rad_dist[j] = funcs.dist(subhalo_position,gas_pos[j],boxSize)
        if not onlyGas:
            for s in nb.prange(star_pos.shape[0]):
                rad_dist[gas_pos.shape[0] + s] = funcs.dist(subhalo_position,star_pos[s],boxSize)

        max_dist = num_grc200 * groupRcrit200[extrapolated_sub_ids[index]]
#         if(rad_dist.size == 1):
#             print(sub_ids[extrapolated_sub_ids[index]], end=' ')
        profile = distance_binning(rad_dist,num_bins, max_dist)

        if index == 0:
            mass_rad_profile = rad_dist
        else:
            mass_rad_profile = np.concatenate((mass_rad_profile,rad_dist))

        if rad_dist.size > 0:
            sub_means[index] = np.mean(rad_dist)
        else:
            sub_means[index] = np.nan

        profiles[index,:] = profile
#     mass_rad_profile = mass_rad_profile[np.where(mass_rad_profile<boxSize)[0]]           
    mass_mean = np.nanmean(mass_rad_profile)
    
    return mass_mean, profiles, sub_means

def AllTracerProfile_wMassBins(basePath, start_snap, target_snap, sub_ids, groupRcrit200, numBins, num_grc200, boxSize):
    onlyGas = False
    header = il.groupcat.loadHeader(basePath,target_snap)
    redshift = header['Redshift']
    h_const = header['HubbleParam']
    boxSize = header['BoxSize']
    
    file = 'files/'+basePath[32:39]+'/all_parent_indices.hdf5'
    if isfile(file):
        f = h5py.File(file,'r')
    else:
        f = h5py.File(f'/vera/ptmp/gc/olwitt/TNG50-1/parent_indices_{target_snap}.hdf5','r')
    
    sub_positions = h5py.File('files/'+basePath[32:39]+'/SubhaloPosAtAllSnaps_extrapolated.hdf5','r') 
    is_extrapolated = sub_positions['is_extrapolated'][:]
    
    #Filter out halos without any subhalo
    sub_ids = sub_ids[np.where(sub_ids != -1)]
    
    extrapolated_sub_ids = np.where(is_extrapolated[sub_ids])[0]
    
#     new_is_extrapolated = is_extrapolated[:]
#     is_extrapolated = new_is_extrapolated.copy()
#     del new_is_extrapolated
    
    sub_pos_at_target_snap = sub_positions['SubhaloPos'][:,:,:] / h_const
#     new_sub_pos = sub_pos_at_target_snap[:]
#     sub_pos_at_target_snap = new_sub_pos.copy()
#     del new_sub_pos
    sub_positions.close()
    
    parent_indices = f[f'snap_0{target_snap}/parent_indices'][:,:]
    parent_indices_data = parent_indices[:,:].astype(int)
    
    if f.__contains__(f'snap_0{target_snap}/numTracersInParents'):
        numTracersInParents = f[f'snap_0{target_snap}/numTracersInParents'][:]
    else:
        numTracersInParents = f[f'snap_0{target_snap}/tracers_in_parents_offset'][:]
    f.close()
    
    all_gas_pos = il.snapshot.loadSubset(basePath,target_snap,'gas',fields=['Coordinates']) / h_const
    if not onlyGas:
        all_star_pos = il.snapshot.loadSubset(basePath,target_snap,'stars',fields=['Coordinates']) / h_const
    
    insituStarsInSubOffset = tF.insituStarsInSubOffset(basePath,start_snap)
    
    #there might be more tracers -> parents in one galaxy at higher redshifts than insitu stars at redshift 0
    final_offsets = tF.tracersInSubhalo(insituStarsInSubOffset,numTracersInParents).astype(int)
    final_offsets = np.insert(final_offsets,0,0)
    
    del insituStarsInSubOffset, numTracersInParents
    
    #test, which galaxies have zero tracers of insitu stars -> those won't have a meaningful radial profile
    # (this already excludes all galaxies without any stars, since they can't have insitu stars)
    isGalaxy = np.full(extrapolated_sub_ids.shape[0], False)
    for i in range(extrapolated_sub_ids.shape[0]):
        sub_id = sub_ids[extrapolated_sub_ids[i]]
        if final_offsets[sub_id + 1] - final_offsets[sub_id] < 2:
            continue
        isGalaxy[i] = True
    
    #only use galaxies that have at least one tracer particle AND have an extrapolated SubhaloPos entry
    extrapolated_sub_ids = extrapolated_sub_ids[np.where(isGalaxy)]
    del isGalaxy
    
    mass_mean, profiles, sub_means =\
    mass_bin_profiles(parent_indices_data, final_offsets, all_gas_pos, all_star_pos, onlyGas, start_snap,\
                      target_snap, sub_pos_at_target_snap, extrapolated_sub_ids, numBins, sub_ids, groupRcrit200,\
                      num_grc200)
    
    return mass_mean, profiles, sub_means, extrapolated_sub_ids

#---- settings----#
halo_type = sys.argv[1]
run = int(sys.argv[2])
target_snap = int(sys.argv[3])

basePath='/virgotng/universe/IllustrisTNG/TNG50-' + str(run) + '/output'
start_snap = 99
#snaps = np.array([99, 84, 67, 50, 33, 25])


h_const = il.groupcat.loadHeader(basePath,99)['HubbleParam']
boxSize = il.groupcat.loadHeader(basePath,99)['BoxSize']

#introduce mass bins:
groups = il.groupcat.loadHalos(basePath, start_snap, fields = ['Group_M_Crit200','GroupFirstSub','Group_R_Crit200'])
group_masses = groups['Group_M_Crit200']*1e10/h_const

#differentiate between halos of dwarf / milky way / group size
dwarf_ids = np.where(np.logical_and(group_masses > 10**(10.8), group_masses < 10**(11.2)))[0]
mw_ids = np.where(np.logical_and(group_masses > 10**(11.8), group_masses < 10**(12.2)))[0]
group_ids = np.where(np.logical_and(group_masses > 10**(12.6), group_masses < 10**(13.4)))[0]
all_ids = np.arange(group_masses.shape[0])

#find ids of associated centrals
sub_ids_dwarves = groups['GroupFirstSub'][dwarf_ids]
sub_ids_mw = groups['GroupFirstSub'][mw_ids]
sub_ids_groups = groups['GroupFirstSub'][group_ids]
all_central_ids = groups['GroupFirstSub'][:]

numBins = 501
num_gmcrit200 = 5
dist_bins = np.linspace(0,num_gmcrit200,numBins)

sub_positions = h5py.File('files/'+basePath[32:39]+'/SubhaloPosAtAllSnaps_extrapolated.hdf5','r') 
is_extrapolated = sub_positions['is_extrapolated'][:]
sub_positions.close()

sub_ids = sub_ids_mw if halo_type == 'mw' else sub_ids_dwarves if halo_type == 'dwarves' else sub_ids_groups if halo_type == 'groups' else all_central_ids

ids = mw_ids if halo_type == 'mw' else dwarf_ids if halo_type == 'dwarves' else group_ids if halo_type == 'groups' else all_ids

group_R_crit200 = groups['Group_R_Crit200'][ids] / h_const

mass_mean, profiles, sub_means, extrapolated_sub_ids = AllTracerProfile_wMassBins(basePath,start_snap, target_snap, sub_ids,\
                                                                                 group_R_crit200, numBins,num_gmcrit200, boxSize)

cumsum_profiles = np.cumsum(profiles, axis = 1)
# profile_cumsum_evo = np.empty((snaps.shape[0],extrapolated_sub_ids.shape[0], numBins), dtype = float)
# mass_means = np.empty(snaps.shape[0], dtype = float)
# sub_means_evo = np.empty((snaps.shape[0], extrapolated_sub_ids.shape[0]), dtype = float)

# mass_means[0] = mass_mean
# profile_cumsum_evo[0,:,:] = np.cumsum(profiles,axis=1)
# sub_means_evo[0,:] = sub_means

# for i in range(1,snaps.shape[0]):
#     mass_mean, profiles, sub_means, _ = AllTracerProfile_wMassBins(basePath,start_snap,snaps[i], sub_ids,\
#                                                                                  group_R_crit200, numBins,num_gmcrit200, boxSize)
#     mass_means[i] = mass_mean
#     profile_cumsum_evo[i,:,:] = np.cumsum(profiles,axis=1)
#     sub_means_evo[i,:] = sub_means
    
f = h5py.File('/vera/ptmp/gc/olwitt/'+basePath[32:39]+'/cumulative_radial_profile/rad_prof_tracer_frac_' + halo_type + f'_{target_snap}.hdf5','w')
ds = f.create_dataset('distance_bins',data = dist_bins)
#ds2 = f.create_dataset('snapshots',data = snaps)
ds3 = f.create_dataset('mass_bin_means',data = mass_means)
ds4 = f.create_dataset('cumulative_profiles',data = cumsum_profiles)
ds5 = f.create_dataset('subhalo_mean_distances',data = sub_means)
ds5 = f.create_dataset('which_galaxy_ids', data = extrapolated_sub_ids)
f.close()

