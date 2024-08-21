import matplotlib.pyplot as plt
import numpy as np
import h5py
import illustrisFuncs as iF
import matplotlib as mpl
from os.path import isdir, isfile

import sys

plt.style.use('fancy_plots2.mplstyle')

def one_gal_profile_from_radial_profile(basePath, sub_id, start_snap, end_snap):
    snaps = np.arange(start_snap, end_snap,-1)
    n = snaps.size
    dist_bins = np.linspace(0,1.5,201)
    profile_evo = np.empty((n,dist_bins.size))
    for i in range(n):
        assert isfile(f'/vera/ptmp/gc/olwitt/insitu/{basePath[32:39]}/lagrangian_regions/lagrangian_regions_w_profiles_{snaps[i]}_single.hdf5'),\
            'Profile file does not exist!'
        filename = '/vera/ptmp/gc/olwitt/insitu/' + basePath[32:39] +\
                      f'/lagrangian_regions/lagrangian_regions_w_profiles_{snaps[i]}_single.hdf5'
        f = h5py.File(filename,'r')
        # change to shmr profile if needed
        profiles = f[f'cumulative_radial_profiles_r_vir'][:,:]
        profile_evo[i] = profiles[0,:]
        print(f'snap {snaps[i]} done;',end=' ', flush = True)
    return dist_bins, profile_evo

run = int(sys.argv[1])
basePath = f'/virgotng/universe/IllustrisTNG/TNG50-{run}/output'
start_snap = 99
sub_id = 167392
target_snap = 24

# specify output directory
dirname = 'pics/radial_profiles'
assert isdir(dirname), 'output directory does not exist'

z = iF.give_z_array(basePath)

bins, num = one_gal_profile_from_radial_profile(basePath, sub_id, start_snap, target_snap)

diff = np.diff(num,axis=1,prepend=0)
diff[np.where(np.isnan(diff))] = 1e-5
diff[np.where(diff == 0)] = 1e-5

# plot until snapshot 25 (z=3)
stop = 99 - (target_snap + 1)
fig, ax = plt.subplots(1,1, figsize = (16,9))
plt.pcolormesh(z[:stop],bins[1:],diff[:stop,1:].T,norm = mpl.colors.LogNorm(2e-4,0.1), shading = 'auto',linewidth = 0,\
               rasterized = True)
cb = plt.colorbar(label = r'$M_{\ast, \,\rm in-situ}\,([r,r+\Delta r, z])\,/\, M_{\ast,\,\rm in-situ}\,(z)$')
plt.ylabel(r'radial distance [$R_{\rm 200c}$]')
plt.xlabel('redshift')
plt.tight_layout()
plt.savefig(dirname + f'/rad_prof_tracer_frac_group_TNG50-{run}_1.5r_vir.pdf',format='pdf')