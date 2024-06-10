import numpy as np
import h5py
import sys
from os.path import isfile, isdir

import tracerFuncs as tF
#
def insitu_catalog(run, stype, start_snap):
    """Decides for every tracer whether it formed in-situ or med-situ. (Future: Updates the RG+16 definitions for central subhalos.)"""
    
    basePath=f'/virgotng/universe/IllustrisTNG/TNG50-{run}/output'
    
    f = h5py.File('/vera/ptmp/gc/olwitt/' + stype + f'/TNG50-{run}/star_formation_snapshots.hdf5','r')
    star_formation_snaps = f['star_formation_snapshot'][:]
    num_tracers = star_formation_snaps.shape[0]
    f.close()
    
    file = '/vera/ptmp/gc/olwitt/' + stype + f'/TNG50-{run}/infall_and_leaving_times.hdf5'
    f = h5py.File(file,'r')
    galaxy_infall_snaps = f['galaxy_infall'][:]
    halo_infall_snaps = f['halo_infall'][:]
    f.close()
    
    insitu_cat = np.zeros(num_tracers, dtype = np.ubyte)
    
    #med-situ stars: either formed before infall into galaxy or never entered galaxy at all (but star formed after infall into halo)
    medsitu = np.where(np.logical_or(np.logical_and(star_formation_snaps < galaxy_infall_snaps, galaxy_infall_snaps != -1),\
                                     np.logical_and(galaxy_infall_snaps == -1, star_formation_snaps > halo_infall_snaps)))[0]
    
    insitu_cat[medsitu] = 1
    
    f = h5py.File('/vera/ptmp/gc/olwitt/' + stype + f'/TNG50-{run}/parent_indices_{start_snap}.hdf5','r')
    numTracersInParents = f[f'snap_{start_snap}/numTracersInParents'][:]
    f.close()
    insituStarsInSubOffset = tF.insituStarsInSubOffset(basePath,start_snap)
    final_offsets = tF.tracersInSubhalo(insituStarsInSubOffset,numTracersInParents).astype(np.int32)
    final_offsets = np.insert(final_offsets,0,0)
        
    file = '/vera/ptmp/gc/olwitt/' + stype + f'/TNG50-{run}/lagrangian_regions/lagrangian_regions_99.hdf5'
    f = h5py.File(file,'r')
    subhaloFlag = f['subhaloFlag'][:]
    f.close()
    
    return insitu_cat, final_offsets, subhaloFlag

run = int(sys.argv[1])
stype = 'insitu'
start_snap = 99

cat, offsets, flags = insitu_catalog(run, stype, start_snap)

file = f'/vera/ptmp/gc/olwitt/auxCats/TNG50-{run}/insitu_or_medsitu_{start_snap}.hdf5'
f = h5py.File(file,'w')
f.create_dataset('stellar_assembly', data = cat)
f.create_dataset('subhalo_offsets', data = offsets)
f.create_dataset('subhalo_flag', data = flags)
f.close()
