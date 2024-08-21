import time
import numpy as np
import h5py
import numba as nb
from numba import jit, njit
from os.path import isfile, isdir
import sys
import illustris_python as il
import illustrisFuncs as iF

@jit(nopython=True)#, parallel = True)
def formation_snapshot(hmr, subhaloFlag, snaps, num, z_arr):
    #hmr shape: (snapshots, subs, all/med-situ/in-situ, accretion channels)
    num_subs = hmr.shape[1]
    num_situ_types = hmr.shape[2]
    num_ac_types = hmr.shape[3]
    form_snap = np.zeros((num_subs, num_situ_types, num_ac_types), dtype = np.float32)

    # loop over all/med-situ/in-situ and accretion channels
    for situ_type, ac_type in np.ndindex(hmr.shape[2:]):
        # loop over all galaxies
        for i in range(num_subs):
            if subhaloFlag[i] == 0:
                form_snap[i, situ_type, ac_type] = np.nan
                continue
            ind = np.where(hmr[:, i, situ_type, ac_type] == num)[0]
            if ind.size > 0:
                if ind.size == 1:
                    form_snap[i, situ_type, ac_type] = snaps[ind[0]]
                else:
                    #choose first occurence as formation snapshot (apparently this never happens, but just in case)
                    form_snap[i, situ_type, ac_type] = snaps[ind.flatten()][0]
                continue
            r_dist = num - hmr[:, i, situ_type, ac_type]
            
            if np.all(r_dist > 1):
                form_snap[i, situ_type, ac_type] = np.nan
                continue
            
            ind = np.where(r_dist[1:] * r_dist[:-1] < 0)[0]
            # if there's no match at all, put snapshot 0 => most likely the hmr hasn't reached one yet
            if ind.size == 0:
                form_snap[i, situ_type, ac_type] = np.nan
                continue
    #         if len(ind.shape) >1:
    #             print(i, ind)

            m = (hmr[ind[0] + 1, i, situ_type, ac_type] - hmr[ind[0], i, situ_type, ac_type])/\
                (snaps[ind[0]+1] - snaps[ind[0]])
            form_snap[i, situ_type, ac_type] = (1 - hmr[ind[0], i, situ_type, ac_type])/m + snaps[ind[0]]

    #        m = (hmr[ind[0]+1,i] - hmr[ind[0],i])/(snaps[ind[0]+1] - snaps[ind[0]])
    #        form_snap[i] = (1 - hmr[ind[0],i])/m + snaps[ind[0]]
        tmp = iF.snap_to_z(z_arr, form_snap[:, situ_type, ac_type])
        form_snap[:, situ_type, ac_type] = tmp
    return form_snap

def formation_redshifts(run, stype):
    """Computes the halfmass radii and estimates the formation snapshots (z at which hmr == R_vir)
    for all central galaxies."""

    start = time.time()
    file = '/vera/ptmp/gc/olwitt/' + stype + f'/TNG50-{run}/lagrangian_regions/lagrangian_regions_99.hdf5'
    assert isfile(file), 'Lagrangian regions file does not exist!'
    f = h5py.File(file,'r')
    arr = f['lagrangian_regions_r_vir'][:,:,:] #(sub_id,regular/igm/mergers), (all tracers, med-situ, in-situ)
    subhaloFlag = f['subhaloFlag'][:]
    f.close()

    z_arr = iF.give_z_array(f'/virgotng/universe/IllustrisTNG/TNG50-{run}/output')
    snaps = np.arange(99,-1,-1)

    hmr_shmr = np.zeros((snaps.shape[0], arr.shape[0], arr.shape[1], arr.shape[2]),\
                         dtype = arr.dtype)
    hmr_r_vir = hmr_shmr.copy()
    tmp_snaps = snaps.copy()
    del_counter = 0
    for i in range(snaps.size):
        file = '/vera/ptmp/gc/olwitt/' + stype + f'/TNG50-{run}/lagrangian_regions/lagrangian_regions_{snaps[i]}.hdf5'
        assert isfile(file), f'Lagrangian regions file at snapshot {snaps[i]} does not exist!'
        f = h5py.File(file,'r')
        if f['lagrangian_regions_shmr'].shape[0] == 1:
            tmp_snaps = np.delete(tmp_snaps, i - del_counter)
            del_counter += 1
            continue
        sub_medians_shmr = f['lagrangian_regions_shmr'][:,:,:]
        sub_medians_r_vir = f['lagrangian_regions_r_vir'][:,:,:]
        f.close()

        hmr_shmr[i,:,:,:] = sub_medians_shmr
        hmr_r_vir[i,:,:,:] = sub_medians_r_vir
        
        print('snap ',snaps[i],' loaded;',flush=True,end=' ')
    del sub_medians_shmr, sub_medians_r_vir
    
    print('\n',tmp_snaps.size, ' snapshots loaded')
    print('# of deleted snapshots: ',del_counter)
    snaps = tmp_snaps
    del tmp_snaps, del_counter

    end_loading = time.time()
    print('loading complete in ',end_loading-start)
    
    num_r_vir = 1
    num_shmr = 2
    
    form_z_shmr = formation_snapshot(hmr_shmr, subhaloFlag, snaps, num_shmr, z_arr)
    form_z_r_vir = formation_snapshot(hmr_r_vir, subhaloFlag, snaps, num_r_vir, z_arr)

    end_form_snap = time.time()
    print('formation snapshot computation complete in ',end_form_snap-end_loading)
    
    assert isdir(f'/vera/ptmp/gc/olwitt/auxCats/TNG50-{run}'), 'Directory does not exist!'
    f = h5py.File(f'/vera/ptmp/gc/olwitt/auxCats/TNG50-{run}/sub_form_redshifts.hdf5','w')
    f.create_dataset('formation_redshifts_shmr',data = form_z_shmr)
    f.create_dataset('formation_redshifts_r_vir',data = form_z_r_vir)
    f.close()
    return

run = int(sys.argv[1])
stype = str(sys.argv[2])
formation_redshifts(run, stype)