import numpy as np
import h5py
import sys
import time
import illustris_python as il
import numba as nb
from numba import jit, njit
import tracerFuncs as tF
import locatingFuncs as lF
import funcs

def one_gal_profile_vs_redshift(basePath, start_snap, subID = 0, max_dist = 800):
    from os.path import isfile
    
    onlyGas = False
    start = time.time()
    header = il.groupcat.loadHeader(basePath,start_snap)
    boxSize = header['BoxSize']
    
    file = 'files/'+basePath[32:39]+'/all_parent_indices.hdf5'
    if isfile(file):
        f = h5py.File('files/'+basePath[32:39]+'/all_parent_indices.hdf5','r')
    else:
        f = h5py.File('/vera/ptmp/gc/olwitt/'+basePath[32:39]+f'/parent_indices_{start_snap}.hdf5','r')
    numTracersInParents = f[f'snap_0{start_snap}/numTracersInParents'][:]
    snaps = np.arange(start_snap,1,-1)
    
    sub_positions = h5py.File('files/' + basePath[32:39] + '/SubhaloPosAtAllSnaps_extrapolated.hdf5','r') 
    sub_pos = sub_positions['SubhaloPos'][subID,:,:]
    is_extrapolated = sub_positions['is_extrapolated'][subID]
    assert is_extrapolated, 'choose a halo with extrapolated subhalo position history'
    sub_positions.close()
    
    end_load = time.time()
    print('time for loading: ',end_load-start)
    
    insituStarsInSubOffset = tF.insituStarsInSubOffset(basePath,start_snap)
    #there might be more tracers -> parents in one galaxy at higher redshifts than insitu stars at redshift 0
    parentsInSubOffset = tF.tracersInSubhalo(insituStarsInSubOffset,numTracersInParents).astype(int)
    parentsInSubOffset = np.insert(parentsInSubOffset,0,0)
    
    end_offsets = time.time()
    print('time for offsets: ',end_offsets-end_load)
    
    num_bins = 100
    bins = np.linspace(0,max_dist,num_bins)
    num = np.zeros((snaps.size,num_bins))
    tree = il.sublink.loadTree(basePath,start_snap,subID,fields = ['SubfindID'],onlyMPB = True)
    
    for i in range(snaps.size):
    #for i in range(1):
        file = 'files/' + basePath[32:39] + '/all_parent_indices.hdf5'
        if isfile(file):
            all_parent_indices = h5py.File(file,'r')
        else:
            all_parent_indices = h5py.File('/vera/ptmp/gc/olwitt/' + basePath[32:39] + f'/parent_indices_{snaps[i]}.hdf5','r')
        parent_indices = all_parent_indices[f'snap_0{snaps[i]}/parent_indices']\
        [parentsInSubOffset[subID]:parentsInSubOffset[subID+1],:]
        
        all_gas_pos = il.snapshot.loadSubset(basePath, snaps[i], 'gas', fields = ['Coordinates'])
        gas_parent_indices = parent_indices[np.where(parent_indices[:,1] == 0)[0],0]
        gas_pos = all_gas_pos[gas_parent_indices.astype(int)]
        if not onlyGas:
            all_star_pos = il.snapshot.loadSubset(basePath, snaps[i], 'stars', fields = ['Coordinates'])
            star_parent_indices = parent_indices[np.where(parent_indices[:,1] == 1)[0],0]
            star_pos = all_star_pos[star_parent_indices.astype(int)]
        
        end_load2 = time.time()
        #print('time for 2nd loading: ',end_load2-end_offsets)

        subhalo_position = sub_pos[99-snaps[i]]
        
        rad_dist = np.zeros(gas_pos.shape[0])
        if not onlyGas:
            rad_dist = np.zeros(gas_pos.shape[0] + star_pos.shape[0])
        for j in range(gas_pos.shape[0]):
            rad_dist[j] = funcs.dist(subhalo_position,gas_pos[j],boxSize)
        if not onlyGas:
            for j in range(star_pos.shape[0]):
                rad_dist[gas_pos.shape[0] + j] = funcs.dist(subhalo_position,star_pos[j],boxSize)
        
        end_prof = time.time()
        #print('time for computing profiles: ',end_prof-end_load2)
        
        num[i,:] = funcs.binData_w_bins(rad_dist,bins = bins)
        num[i,:] = num[i,:]/sum(num[i,:]) #we're only interested in the fraction of tracers in that bin
        
        end_bin = time.time()
        #print('time for binning: ',end_bin-end_prof)
    
        print(snaps[i],' done;',end=' ',flush=True)
    all_parent_indices.close()
    return bins, num

run = int(sys.argv[1])
basePath = '/virgotng/universe/IllustrisTNG/TNG50-' + str(run) + '/output'
subID = int(sys.argv[2])
max_dist = 800
bins, num = one_gal_profile_vs_redshift(basePath,99, subID, max_dist)

f = h5py.File('files/' + basePath[32:39] + f'/rad_prof_z_tracer_frac_sub{subID}_{max_dist}.hdf5','w')
ds = f.create_dataset('bins',bins.shape,dtype = float)
ds[:] = bins
ds3 = f.create_dataset('values',num.shape,dtype = float)
ds3[:] = num
f.close()