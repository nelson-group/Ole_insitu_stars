import numpy as np
import h5py
import sys

def find_star_formation_snap(run, stype, start_snap, num_tracers):
    """Computes snapshot of star formation for every tracer based on the parent index tables."""
    
    snaps = np.arange(start_snap, -1,-1)
    star_formation_snap = np.full(num_tracers, -1, dtype = np.byte)
    
    for i, snap in enumerate(snaps):
        file = '/vera/ptmp/gc/olwitt/' + stype + f'/TNG50-{run}/parent_indices_{snap}.hdf5'
        f = h5py.File(file,'r')
        #only load information of state of tracer parent (gas/star)
        new_parent_indices = f[f'snap_{snap}/parent_indices'][:,1]
        f.close()
        
        if i > 0:
            #save last snapshot of star formation, i.e. the highest snapshot at which the gas parent turns into a star
            new_star = np.where(np.logical_and(np.logical_and(new_parent_indices == 0, old_parent_indices == 1), star_formation_snap == -1))
            star_formation_snap[new_star] = snap + 1
        
        old_parent_indices = new_parent_indices.copy()
    
    return star_formation_snap

run = int(sys.argv[1])
stype = str(sys.argv[2])
start_snap = 99

file = '/vera/ptmp/gc/olwitt/' + stype + f'/TNG50-{run}/parent_indices_{start_snap}.hdf5'
f = h5py.File(file,'r')
#only load information of state of tracer parent (gas/star)
new_parent_indices = f[f'snap_{start_snap}/parent_indices'][:,1]
num_tracers = new_parent_indices.shape[0]
del new_parent_indices
f.close()

star_formation_snaps = find_star_formation_snap(run, stype, start_snap, num_tracers)

file = '/vera/ptmp/gc/olwitt/' + stype + f'/TNG50-{run}/star_formation_snapshots.hdf5'
f = h5py.File(file,'w')
f.create_dataset('star_formation_snapshot', data = star_formation_snaps)
f.close()
