import time
import illustris_python as il
import numpy as np
import h5py
import numba as nb
from numba import jit, njit
import tracerFuncs as tF
import funcs
import illustrisFuncs as iF
import sys
from os.path import isfile, isdir

@jit(parallel = True, nopython = True)
def tracer_sub_distance_computation(tracer_pos, sub_positions, sub_shmr, tracersInSubOffset, subhaloFlag, boxSize):
    tracer_distances = np.zeros(tracer_pos.shape[0])
    for i in nb.prange(sub_positions.shape[0]):
        if subhaloFlag[i] == 0:
            continue
        tracersInSub = np.arange(tracersInSubOffset[i],tracersInSubOffset[i+1])
        tracer_distances[tracersInSub] = funcs.dist_vector_nb(sub_positions[i], tracer_pos[tracersInSub], boxSize)
        # normalize distances to stellar halfmass radius
        tracer_distances[tracersInSub] = tracer_distances[tracersInSub] / sub_shmr[i] 
    return tracer_distances

@jit(nopython = True, parallel = True)
def binning(distances, time_means, igm_mask, satellite_mask, tracersInSubOffset, subhaloFlag, numBins, max_dist): 
    numSubs = tracersInSubOffset.shape[0] - 1
    profiles = np.zeros((numSubs, 3, numBins))
    medians = np.zeros((numSubs, 3))
    
    minVal = 0
    maxVal = max_dist
    binWidth = (maxVal - minVal) / numBins
    for i in nb.prange(numSubs):
        if subhaloFlag[i] == 0:
            continue
        tracersInSub = np.arange(tracersInSubOffset[i],tracersInSubOffset[i+1])
        igm_tracers = tracersInSub[igm_mask[tracersInSub]]
        satellite_tracers = tracersInSub[satellite_mask[tracersInSub]]
        
        regular_med = np.full(numBins,np.nan)
        igm_med = np.full(numBins,np.nan)
        satellite_med = np.full(numBins,np.nan)
        
        medians[i,0] = np.nanmedian(time_means[tracersInSub])
        medians[i,1] = np.nanmedian(time_means[igm_tracers])
        medians[i,2] = np.nanmedian(time_means[satellite_tracers])
        
        for j in nb.prange(numBins):
            relInd = np.where(np.logical_and(distances[tracersInSub] >= minVal + j*binWidth, distances[tracersInSub] < minVal +\
                                             (j+1)*binWidth))[0]
            if(relInd.shape[0] > 0):
                regular_med[j] = np.nanmedian(time_means[tracersInSub[relInd]])
    
            relInd = np.where(np.logical_and(distances[igm_tracers] >= minVal + j*binWidth, distances[igm_tracers] < minVal +\
                                             (j+1)*binWidth))[0]
            if(relInd.shape[0] > 0):
                igm_med[j] = np.nanmedian(time_means[igm_tracers[relInd]])
    
            relInd = np.where(np.logical_and(distances[satellite_tracers] >= minVal + j*binWidth, distances[satellite_tracers] < minVal +\
                                             (j+1)*binWidth))[0]
            if(relInd.shape[0] > 0):
                satellite_med[j] = np.nanmedian(time_means[satellite_tracers[relInd]])
            
        profiles[i,0,:] = regular_med
        profiles[i,1,:] = igm_med
        profiles[i,2,:] = satellite_med
    return profiles, medians

def igm_to_sub_times(basePath, stype, num_shmr, num_bins, start_snap, special_sub_ids):
    """For every in-situ star particle in all subhalos, this function returns the average redshift
    at which its tracers entered a galaxy for the first (last) time. Afterward, a radial profile of this is computed."""
    start = time.time()
    h_const = il.groupcat.loadHeader(basePath, start_snap)['HubbleParam']
    boxSize = il.groupcat.loadHeader(basePath, start_snap)['BoxSize']
    
    run = basePath[38]
    
    z = iF.give_z_array(basePath)   

    #necessary offsets, when not every tracer is important:
    if stype == 'insitu':
        insituStarsInSubOffset = tF.insituStarsInSubOffset(basePath,start_snap)
    elif stype == 'exsitu':
        insituStarsInSubOffset = tF.exsituStarsInSubOffset(basePath,start_snap)
    else:
        raise Exception('Invalid star type!')
    
    assert isfile('/vera/ptmp/gc/olwitt/' + stype + '/' + basePath[32:39] + f'/parent_indices_{start_snap}.hdf5'),\
        'parent indices file does not exist!'
    file = '/vera/ptmp/gc/olwitt/' + stype + '/' + basePath[32:39] + f'/parent_indices_{start_snap}.hdf5'
    f = h5py.File(file,'r')
    parent_indices = f[f'snap_{start_snap}/parent_indices'][:]
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
    
    f = h5py.File('/vera/ptmp/gc/olwitt/auxCats' + basePath[32:39] + '/subhaloFlag_' + stype + '.hdf5','r')
    subhaloFlag = f['subhaloFlag'][:]
    f.close()
    
    # load accretion channels:
    assert isfile('/vera/ptmp/gc/olwitt/auxCats/' + basePath[32:39] + f'/tracer_accretion_channels_{start_snap}.hdf5'),\
        'accretion channels file does not exist!'
    f = h5py.File('/vera/ptmp/gc/olwitt/auxCats/' + basePath[32:39] + f'/tracer_accretion_channels_{start_snap}.hdf5','r')
    accretion_channels = f['tracer_accretion_channels'][:]
    f.close()
    igm = np.where(funcs.isin(accretion_channels,np.array([0,1])))[0]
    igm_mask = np.full(num_tracers,False)
    igm_mask[igm] = True

    satellite = np.where(accretion_channels == 2)[0]
    satellite_mask = np.full(num_tracers,False)
    satellite_mask[satellite] = True
    
    # print(num_tracers, np.nonzero(igm_mask)[0].shape[0], np.nonzero(satellite_mask)[0].shape)
    
    #load halo infall time for every tracer:
    assert isfile('/vera/ptmp/gc/olwitt/' + stype + f'/TNG50-{run}/infall_and_leaving_times.hdf5'),\
        'infall and leaving times file does not exist!'
    f = h5py.File('/vera/ptmp/gc/olwitt/' + stype + f'/TNG50-{run}/infall_and_leaving_times.hdf5','r')
    halo_infall = f['halo_infall'][:]
    f.close()
    
    #load relevant star positions at z=0:
    file = '/vera/ptmp/gc/olwitt/' + stype + '/' + basePath[32:39] + f'/parent_indices_{start_snap}.hdf5'
    if not isfile(file):
        file = 'files/' + basePath[32:39] + '/all_parent_indices.hdf5'
        assert isfile(file), 'parent indices file does not exist!'
    f = h5py.File(file,'r')
    parent_indices = f[f'snap_{start_snap}/parent_indices'][:,:]
    parent_indices = parent_indices[:,0].astype(int)
    f.close()
    
    end_loading = time.time()
    print('loading completed in ',end_loading - start,flush=True)

    end_loop = time.time()
    print('\nloop complete in ',end_loop - end_loading, flush = True)
    
    star_pos_insitu = il.snapshot.loadSubset(basePath, start_snap, 4, fields = ['Coordinates'])[parent_indices,:] / h_const
    
    #load subhalo positions for distance computation
    sub_positions = il.groupcat.loadSubhalos(basePath, start_snap, fields = ['SubhaloPos'])[:,:] / h_const
    sub_shmr = il.groupcat.loadSubhalos(basePath, start_snap, fields = ['SubhaloHalfmassRadType'])[:,4] / h_const
    
    mask = np.where(halo_infall != -1)[0]
    time_means = np.full(num_tracers, np.nan)
    time_means[mask] = z[99-halo_infall[mask]]

    
    end_time_means = time.time()
    print('time means complete in ',end_time_means - end_loop, flush = True)
    
    tracer_distances = tracer_sub_distance_computation(star_pos_insitu, sub_positions, sub_shmr, parentsInSubOffset, subhaloFlag, boxSize)
    end_distances = time.time()
    print('distances complete in ',end_distances - end_time_means, flush = True)
    assert time_means.shape[0] == tracer_distances.shape[0]
    
    # print(sub_positions[special_sub_ids])

    # save distances and time means for special subhalos for plotting purposes
    special_time_means = {}
    special_tracer_pos = {}
    special_igm = {}
    special_satellite = {}
    for i in range(special_sub_ids.size):
        print('sub_id: ', special_sub_ids[i])
        tracers = np.arange(parentsInSubOffset[special_sub_ids[i]],\
                            parentsInSubOffset[special_sub_ids[i] + 1])

        igm_tracers = tracers[igm_mask[tracers]]
        if igm_tracers.size > 0:
            igm_tracers -= igm_tracers[0]
        else:
            igm_tracers = np.full(1,np.nan)
            
        satellite_tracers = tracers[satellite_mask[tracers]]
        if satellite_tracers.size == tracers.size:
            print('I doubt it!')
        if satellite_tracers.size > 0:
            satellite_tracers -= satellite_tracers[0]
        else:
            satellite_tracers = np.full(1,np.nan)
            
        special_time_means[special_sub_ids[i]] = time_means[tracers]
        special_tracer_pos[special_sub_ids[i]] = star_pos_insitu[tracers]
        special_igm[special_sub_ids[i]] = igm_tracers
        special_satellite[special_sub_ids[i]] = satellite_tracers

    #create radial profiles for each subhalo
    dist_bins = np.linspace(0, num_shmr, num_bins)
    
    profiles, medians = binning(tracer_distances, time_means, igm_mask, satellite_mask, parentsInSubOffset, subhaloFlag, num_bins, num_shmr)
    
    end = time.time()
    print('profiles completed in ',end - end_time_means, flush = True)
    print('finished in ',end - start, flush = True)
    
    return dist_bins, profiles, medians, special_time_means, special_tracer_pos, special_igm, special_satellite

#---- main ----#
run = int(sys.argv[1])
basePath=f'/virgotng/universe/IllustrisTNG/TNG50-{run}/output'
start_snap = 99
# set number of bins and the maximum number of stellar halfmass radii to consider for the radial profiles
num_shmr = 7
num_bins = 15
stype = str(sys.argv[2])

# only for finding suitable special subhalos for plotting
# file = '/vera/ptmp/gc/olwitt/' + stype + f'/TNG50-{run}/lagrangian_regions/lagrangian_regions_99.hdf5'
# f = h5py.File(file,'r')
# subhaloFlag = f['subhaloFlag'][:]
# dwarf_inds = f['mass_bin_sub_ids/dwarfs'][:]
# mw_inds = f['mass_bin_sub_ids/mws'][:]
# f.close()

# dwarf_sub_ids = dwarf_inds[np.nonzero(np.isin(dwarf_inds, np.nonzero(subhaloFlag)[0]))[0]]

# ids1 = np.insert(dwarf_sub_ids,[0,0],[167392,552879])
ids1 = np.array([167392, 552879, 706460])

special_sub_ids = ids1 if run == 1 else\
                              np.array([28044,75514,108365]) if run == 2 else np.array([0,5,10])
dist_bins, profiles, medians, time_means, tracer_pos, igm_tracers, satellite_tracers = igm_to_sub_times(basePath, stype, num_shmr, num_bins, start_snap, special_sub_ids)

#specify path to your directory
assert isdir('files/' + basePath[32:39] + '/mean_infall_times'), 'directory does not exist!'
f = h5py.File('files/' + basePath[32:39] + f'/mean_infall_times/mean_infall_times_all_subs_{num_shmr}shmr.hdf5','w')
f.create_dataset('distance_bins', data = dist_bins)
f.create_dataset('mean_infall_time_profiles', data = profiles)
f.create_dataset('mean_infall_times_medians', data = medians)
for ind in special_sub_ids:
    g = f.create_group(f'special_sub_{ind}')
    g.create_dataset('time_means',data = time_means[ind])
    g.create_dataset('tracer_pos',data = tracer_pos[ind])
    g.create_dataset('tracers_from_igm',data = igm_tracers[ind])
    g.create_dataset('tracers_from_mergers',data = satellite_tracers[ind])
f.close()