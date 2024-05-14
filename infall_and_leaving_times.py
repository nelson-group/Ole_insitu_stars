import numpy as np
import h5py
import sys
from os.path import isfile, isdir

def crossing_snapshots(run, stype, start_snap):
    """Computes snapshot for every tracer, at which R_vir, or 2 R_0.5,star respectively, is crossed."""
    
    snaps = np.arange(start_snap, -1,-1)
    
    f = h5py.File('/vera/ptmp/gc/olwitt/' + stype + f'/TNG50-{run}/star_formation_snapshots.hdf5','r')
    star_formation_snaps = f['star_formation_snapshot'][:]
    num_tracers = star_formation_snaps.shape[0]
    f.close()
    del star_formation_snaps
    
    halo_infall = np.full(num_tracers, -1, dtype = np.byte)
    galaxy_infall = np.full(num_tracers, -1, dtype = np.byte)
    always_inside_gal = np.ones(num_tracers, dtype = np.byte)
    always_inside_halo = np.ones(num_tracers, dtype = np.byte)
    always_outside_galaxy = np.ones(num_tracers, dtype = np.byte)
    
    for i, snap in enumerate(snaps):
        f = h5py.File('/vera/ptmp/gc/olwitt/' + stype + f'/TNG50-{run}/lagrangian_regions/lagrangian_regions_cut21_{snap}.hdf5','r')
        new_inside_galaxy = f['tracers_inside_galaxy'][:]
        new_inside_halo = f['tracers_inside_halo'][:]
        f.close()
        
        # one entry only when no subs/halos are available at that snapshot
        if new_inside_galaxy.shape[0] == 1 and new_inside_galaxy[0] == -1:
            continue
        
        #inside every snapshot so far: delete the ones that are not in the galaxy/halo anymore
        always_inside_gal[np.where(new_inside_galaxy == 0)] = 0
        always_inside_halo[np.where(new_inside_halo == 0)] = 0
        always_outside_galaxy[np.where(new_inside_galaxy == 1)] = 0
        
        if i > 0:
            # find latest (at highest snapshot) galaxy infall time when no other infall has been detected yet (galaxy_infall = -1)
            # -> this is the infall into the galaxy in which they're located at z=0
            new_in_gal = np.where(np.logical_and(np.logical_and(new_inside_galaxy == 0, old_inside_galaxy == 1), galaxy_infall == -1))
            galaxy_infall[new_in_gal] = snap + 1
            
            #same for halo
            new_in_halo = np.where(np.logical_and(np.logical_and(new_inside_halo == 0, old_inside_halo == 1), halo_infall == -1))
            halo_infall[new_in_halo] = snap + 1        
        
        old_inside_galaxy = new_inside_galaxy.copy()
        old_inside_halo = new_inside_halo.copy()
    
    #take stars which formed at snapshot 0 correctly into account (if any)
    galaxy_infall[np.nonzero(always_inside_gal)[0]] == 0
    halo_infall[np.nonzero(always_inside_halo)[0]] == 0
    
    print('# tracers always outside of MP galaxy: ', np.nonzero(always_outside_galaxy)[0].shape[0])
    
    return halo_infall, galaxy_infall

def leaving_igm(run, stype, start_snap):
    """Computes snapshot for every tracer, at which it left the IGM."""
    
    snaps = np.arange(start_snap, -1,-1)
    
    f = h5py.File('/vera/ptmp/gc/olwitt/' + stype + f'/TNG50-{run}/star_formation_snapshots.hdf5','r')
    star_formation_snaps = f['star_formation_snapshot'][:]
    num_tracers = star_formation_snaps.shape[0]
    f.close()
    del star_formation_snaps
    
    leaving_igm_snap = np.full(num_tracers, -1, dtype = np.byte) 
    
    f = h5py.File('/vera/ptmp/gc/olwitt/' + stype + f'/TNG50-{run}/subhalo_index_table.hdf5','r')
    for i, snap in enumerate(snaps):
        
        # skip missing snapshots
        if not f.__contains__(f'snap_{snap}'):
            continue
        
        new_location = f[f'snap_{snap}/location'][:]
        
        
        if i > 0:
            #compute first (i.e. lowest snapshot) snapshot at which tracer leaves the igm
            new_in_gal = np.where(np.logical_and(new_location == -1, old_location != -1))
            leaving_igm_snap[new_in_gal] = snap + 1        
        
        old_location = new_location.copy()
        
        
    f.close()
    return leaving_igm_snap

run = int(sys.argv[1])
stype = str(sys.argv[2])
start_snap = 99

halo_infall, galaxy_infall = crossing_snapshots(run, stype, start_snap)
leaving_igm_snap = leaving_igm(run, stype, start_snap)

file = '/vera/ptmp/gc/olwitt/' + stype + f'/TNG50-{run}/infall_and_leaving_times.hdf5'
f = h5py.File(file,'w')
f.create_dataset('halo_infall', data = halo_infall)
f.create_dataset('galaxy_infall', data = galaxy_infall)
f.create_dataset('leaving_igm', data = leaving_igm_snap)
f.close()
