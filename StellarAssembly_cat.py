import numpy as np
import h5py
import sys
from os.path import isfile, isdir

import funcs
import numba as nb
from numba import jit,njit

@jit(nopython = True, parallel = True)
def find_situ_type_for_every_star(situ, medsitu, numTracersInParents):
    res = situ.copy().astype(np.byte)
    num_insitu_stars = numTracersInParents.shape[0]
    del situ
    insituStarOffsets = funcs.insert(np.cumsum(numTracersInParents),0,0)
    
    insitu = np.where(res == 1)[0]
    
    #only go through in-situ stars again and check their formation type
    for i in nb.prange(num_insitu_stars):
        insitu_star_inds = np.arange(insituStarOffsets[i],insituStarOffsets[i+1])
        #star_type = 0: in-situ, star_type = 1: med-situ
        #add one, bc in-situ is 1 in res and med-situ thus has to be 2
        if insitu_star_inds.shape[0] > 0:
            star_type = int(np.nanmedian(medsitu[insitu_star_inds])) + 1
            res[insitu[i]] = star_type
    
    return res

def insitu_catalog(run, stype, start_snap):
    """Decides for every tracer whether it formed in-situ or med-situ. (Future: Updates the RG+16 definitions for central subhalos.)"""
    
    basePath=f'/virgotng/universe/IllustrisTNG/TNG50-{run}/output'
    
    # catalogs to decide whether star is in-situ or med-situ
    
    f = h5py.File('/vera/ptmp/gc/olwitt/' + stype + f'/TNG50-{run}/star_formation_snapshots.hdf5','r')
    star_formation_snaps = f['star_formation_snapshot'][:]
    f.close()
    
    file = '/vera/ptmp/gc/olwitt/' + stype + f'/TNG50-{run}/infall_and_leaving_times.hdf5'
    f = h5py.File(file,'r')
    galaxy_infall_snaps = f['galaxy_infall'][:]
    halo_infall_snaps = f['halo_infall'][:]
    f.close()
    
    #med-situ stars: either formed before infall into galaxy or never entered galaxy at all (but star formed after infall into halo)
    medsitu = np.where(np.logical_or(np.logical_and(star_formation_snaps < galaxy_infall_snaps, galaxy_infall_snaps != -1),\
                                     np.logical_and(galaxy_infall_snaps == -1, star_formation_snaps > halo_infall_snaps)))[0]
    
    medsitu_tmp = np.zeros(star_formation_snaps.shape[0], dtype = np.byte)
    medsitu_tmp[medsitu] = 1
    medsitu = medsitu_tmp.copy()
    del medsitu_tmp
    
    # -> now we have an array of length num_tracers with 1s where a tracer forms a med-situ star
    
    # offsets for tracers per star
    
    f = h5py.File('/vera/ptmp/gc/olwitt/' + stype + f'/TNG50-{run}/parent_indices_{start_snap}.hdf5','r')
    numTracersInParents = f[f'snap_{start_snap}/numTracersInParents'][:].astype(np.int32)
    f.close()
    
    check = h5py.File(basePath[:-6] + 'postprocessing/StellarAssembly/stars_' + str(start_snap).zfill(3) + '.hdf5','r')
    situ = check['InSitu'][:] #1 if star is formed insitu, 0 if it was formed ex-situ and -1 otherwise (fuzz)
    check.close()
    
    stellar_assembly_cat = find_situ_type_for_every_star(situ, medsitu, numTracersInParents)
        
    file = '/vera/ptmp/gc/olwitt/' + stype + f'/TNG50-{run}/lagrangian_regions/lagrangian_regions_99.hdf5'
    f = h5py.File(file,'r')
    subhaloFlag = f['subhaloFlag'][:]
    f.close()
    
    return stellar_assembly_cat, subhaloFlag

run = int(sys.argv[1])
stype = 'insitu'
start_snap = 99

cat, flags = insitu_catalog(run, stype, start_snap)

file = f'/vera/ptmp/gc/olwitt/auxCats/TNG50-{run}/StellarAssembly_{start_snap}.hdf5'
f = h5py.File(file,'w')
f.create_dataset('stellar_assembly', data = cat)
f.create_dataset('subhalo_flag', data = flags)
f.close()
