import numpy as np
import h5py
import sys
from os.path import isfile, isdir

def gather_star_formation_distances(run, stype, start_snap):
    """Computes distance at star formation w.r.t. to the (MP) subhalo center for every tracer."""
    
    snaps = np.arange(start_snap, -1,-1)
    
    assert isfile('/vera/ptmp/gc/olwitt/' + stype + f'/TNG50-{run}/star_formation_snapshots.hdf5')
    f = h5py.File('/vera/ptmp/gc/olwitt/' + stype + f'/TNG50-{run}/star_formation_snapshots.hdf5','r')
    star_formation_snaps = f['star_formation_snapshot'][:]
    f.close()
    star_formation_dist = np.full(star_formation_snaps.shape[0], -1, dtype = float)
    
    for i, snap in enumerate(snaps):
        assert isfile('/vera/ptmp/gc/olwitt/' + stype + f'/TNG50-{run}/distance_cats/distance_cats_{snap}.hdf5')
        f = h5py.File('/vera/ptmp/gc/olwitt/' + stype + f'/TNG50-{run}/distance_cats/distance_cats_{snap}.hdf5','r')
        dist_at_star_form = f['aux_star_formation_distances'][:]
        f.close()
        new_stars = np.where(star_formation_snaps == snap)[0]
        assert len(new_stars) == len(dist_at_star_form)
        star_formation_dist[new_stars] = dist_at_star_form
    
    return star_formation_dist

run = int(sys.argv[1])
stype = str(sys.argv[2])
start_snap = 99

star_formation_dist = gather_star_formation_distances(run, stype, start_snap)
# print(np.where(star_formation_dist == -1)[0].shape[0])
# print(star_formation_dist.shape[0])

# specify path to your directory
file = '/vera/ptmp/gc/olwitt/' + stype + f'/TNG50-{run}/star_formation_distances.hdf5'
f = h5py.File(file,'w')
f.create_dataset('star_formation_distances', data = star_formation_dist)
f.close()
