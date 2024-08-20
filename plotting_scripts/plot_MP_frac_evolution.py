import matplotlib.pyplot as plt
import numpy as np
import h5py
from os.path import isfile, isdir

plt.style.use('fancy_plots2.mplstyle')

# set index until which you want to plot (until = 87 corresponds to z = 6)
until = 87
run = 1
assert isfile(f'files/TNG50-{run}/accretion_channels_insitu.hdf5'), 'File not found'
f = h5py.File(f'files/TNG50-{run}/accretion_channels_insitu.hdf5','r')
mw_ids = f['subhalo_ids/mws'][:]
dwarf_ids = f['subhalo_ids/dwarfs'][:]
group_ids = f['subhalo_ids/groups'][:]
giant_ids = f['subhalo_ids/giants'][:]

gal_comp = f['galaxy_composition'][:until,:,:]

mp_stars =  f['stars_in_main_progenitor'][:until]
total = f['totals'][:until]
z = f['redshift'][:until]
f.close()

fig, ax = plt.subplots(1,1, figsize = (8,6))
ax.plot(z,mp_stars/total, label = 'all in-situ stars')
ax.plot(z,np.nanmedian(gal_comp[:,dwarf_ids,1], axis = 1), label = 'dwarfs')
ax.plot(z,np.nanmedian(gal_comp[:,mw_ids,1], axis = 1), label = 'MWs')
ax.plot(z,np.nanmedian(gal_comp[:,group_ids,1], axis = 1), label = 'groups')
ax.plot(z,np.nanmedian(gal_comp[:,giant_ids,1], axis = 1), label = 'massive groups')
ax.legend()
ax.set_xlabel('redshift')
ax.ylabel(r'$M_{\ast,\,\rm insitu}\,(z)\,/\, M_{\ast,\,\rm insitu}\,(z=0)$')
fig.tight_layout()

# specify path to your directory to save the plot
dirname = 'pics/tracer_fractions'
assert isdir(dirname), 'Directory not found'
plt.savefig(dirname + f'/insitu_star_number_fraction_50-{run}.pdf',format='pdf')