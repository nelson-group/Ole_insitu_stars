import matplotlib.pyplot as plt
import numpy as np
import h5py
import matplotlib.patches as mpatches
import sys
from os.path import isfile, isdir
plt.style.use('fancy_plots2.mplstyle')

run = int(sys.argv[1])

# specify output directory
dirname = 'pics/tracer_fraction'
assert isdir(dirname), 'output directory does not exist'

# set index until which to plot (z=6)
until = 87

assert isfile(f'/vera/ptmp/gc/olwitt/auxCats/TNG50-{run}/accretion_channels_insitu.hdf5'), 'Accretion channel file does not exist!'
f = h5py.File(f'/vera/ptmp/gc/olwitt/auxCats/TNG50-{run}/accretion_channels_insitu.hdf5','r')
dwarf_ids = f['subhalo_ids/dwarfs'][:]
mw_ids = f['subhalo_ids/mws'][:]
group_ids = f['subhalo_ids/groups'][:]
subhaloFlag = f['subhaloFlag'][:]
z = f['redshift'][:until]
nums = f['binned_values'][:,:,:,0,:] #snap, halo mass bins, baryonic mass bin, value; 0 for all tracers
subs = f['all_galaxies'][:]
totals = f['totals'][:]
f.close()

dwarf_inds = dwarf_ids[np.nonzero(np.isin(dwarf_ids, np.nonzero(subhaloFlag)[0]))[0]]
mw_inds = mw_ids[np.nonzero(np.isin(mw_ids, np.nonzero(subhaloFlag)[0]))[0]]
group_inds = group_ids[np.nonzero(np.isin(group_ids, np.nonzero(subhaloFlag)[0]))[0]]

fig, ax = plt.subplots(2,2,figsize = (16,9))
ax = ax.flatten()
plots = [1,2,3,0]
for i in range(4):
    tsum = np.sum(nums[:until,plots[i],:,3],axis=1)
    y = np.vstack([nums[:until,plots[i],0,3] / tsum, nums[:until,plots[i],1,3] / tsum, nums[:until,plots[i],2,3] / tsum, nums[:until,plots[i],3,3] / tsum, nums[:until,plots[i],4,3] / tsum])
    ax[i].stackplot(z, y, colors = ['purple','blue','green','red','yellow'], alpha = 0.8, edgecolor = 'black')
    ax[i].set_ylim(0,1)
    ax[i].set_xlim(0,6)


purple = mpatches.Patch(color = 'purple', label = r'$\log\,\rm M_{\rm bar} < 9$')
blue = mpatches.Patch(color = 'blue', label = r'$9 < \log\,\rm M_{\rm bar} < 10$')
green = mpatches.Patch(color = 'green', label = r'$10 < \log\,\rm M_{\rm bar} < 11$')
red = mpatches.Patch(color = 'red', label = r'$11 < \log\,\rm M_{\rm bar} < 12$')
yellow = mpatches.Patch(color = 'yellow', label = r'$12 < \log\,\rm M_{\rm bar}$')

ax[0].legend(handles=[purple,blue,green,red,yellow],loc = 'lower right')
ax[0].set_ylabel('mass fraction')
ax[2].set_ylabel('mass fraction')
ax[2].set_xlabel('redshift')
ax[3].set_xlabel('redshift')
ax[3].tick_params(labelleft=False)
ax[1].tick_params(labelleft=False)

ax[0].text(1.4,0.1,'dwarfs', size = 24, bbox=\
             dict(boxstyle="round",ec='black',fc='lightgray',alpha = 0.5))
ax[1].text(1.4,0.1,'MW-like', size = 24, bbox=\
             dict(boxstyle="round",ec='black',fc='lightgray',alpha = 0.5))
ax[2].text(1.4,0.1,'groups', size = 24, bbox=\
             dict(boxstyle="round",ec='black',fc='lightgray',alpha = 0.5))
ax[3].text(1.4,0.1,'all galaxies', size = 24, bbox=\
             dict(boxstyle="round",ec='black',fc='lightgray',alpha = 0.5))
fig.tight_layout()
plt.savefig(dirname + f'/tracer_fraction_TNG50-{run}_onlyOtherGalaxies_stack.pdf',format='pdf')