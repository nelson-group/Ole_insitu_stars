import matplotlib.pyplot as plt
import numpy as np
import h5py
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import sys
from os.path import isfile, isdir

plt.style.use('fancy_plots2.mplstyle')

run = int(sys.argv[1])
# plot until redshift 6
until = 87

# specify output directory
dirname = 'pics/tracer_fraction'
assert isdir(dirname), 'output directory does not exist'

assert isfile(f'/vera/ptmp/gc/olwitt/auxCats/TNG50-{run}/accretion_channels_insitu.hdf5'), 'Accretion channel file does not exist!'
f = h5py.File(f'/vera/ptmp/gc/olwitt/auxCats/TNG50-{run}/accretion_channels_insitu.hdf5','r')
dwarf_ids = f['subhalo_ids/dwarfs'][:]
mw_ids = f['subhalo_ids/mws'][:]
group_ids = f['subhalo_ids/groups'][:]
subhaloFlag = f['subhaloFlag'][:]
gal_comp = f['galaxy_composition'][:,:,0,:]
z0 = f['redshift'][:]
mp1 = f['main_progenitor'][:until]
igm1 = f['IGM'][:until]
sub1 = f['all_galaxies'][:until]
f.close()

dwarf_inds = dwarf_ids[np.nonzero(np.isin(dwarf_ids, np.nonzero(subhaloFlag)[0]))[0]]
mw_inds = mw_ids[np.nonzero(np.isin(mw_ids, np.nonzero(subhaloFlag)[0]))[0]]
group_inds = group_ids[np.nonzero(np.isin(group_ids, np.nonzero(subhaloFlag)[0]))[0]]
 
fig, ax = plt.subplots(1,1, figsize=(16,9))
z = z0

#### all galaxies ####
igm1 = np.nanmedian(1 - gal_comp[:until,np.nonzero(subhaloFlag)[0],0] - gal_comp[:until,np.nonzero(subhaloFlag)[0],3], axis = 1)
igm1_high = np.nanpercentile(1 - gal_comp[:until,np.nonzero(subhaloFlag)[0],0] - gal_comp[:until,np.nonzero(subhaloFlag)[0],3], 84, axis = 1)
igm1_low = np.nanpercentile(1 - gal_comp[:until,np.nonzero(subhaloFlag)[0],0] - gal_comp[:until,np.nonzero(subhaloFlag)[0],3], 16, axis = 1)
plt.plot(z[:until], np.nanmedian(gal_comp[:until,np.nonzero(subhaloFlag)[0],0], axis = 1), color = 'tab:blue')
plt.plot(z[:until], np.nanmedian(gal_comp[:until,np.nonzero(subhaloFlag)[0],3], axis = 1), color = 'tab:orange')
plt.plot(z[:until], igm1, color = 'tab:green')

# plt.fill_between(z[:until], igm1_low, igm1_high, color = 'tab:green', alpha = 0.2)
# plt.fill_between(z[:until], np.nanpercentile(gal_comp[:until,np.nonzero(subhaloFlag)[0],0], 16, axis = 1),\
#                   np.nanpercentile(gal_comp[:until,np.nonzero(subhaloFlag)[0],0], 84, axis = 1), color = 'tab:blue', alpha = 0.2)
# plt.fill_between(z[:until], np.nanpercentile(gal_comp[:until,np.nonzero(subhaloFlag)[0],3], 16, axis = 1),\
#                     np.nanpercentile(gal_comp[:until,np.nonzero(subhaloFlag)[0],3], 84, axis = 1), color = 'tab:orange', alpha = 0.2)

#### dwarfs ####

igm2 = np.nanmedian(1 - gal_comp[:until,dwarf_inds,0] - gal_comp[:until,dwarf_inds,3], axis = 1)
igm2_high = np.nanpercentile(1 - gal_comp[:until,dwarf_inds,0] - gal_comp[:until,dwarf_inds,3], 84, axis = 1)
igm2_low = np.nanpercentile(1 - gal_comp[:until,dwarf_inds,0] - gal_comp[:until,dwarf_inds,3], 16, axis = 1)
plt.plot(z[:until], np.nanmedian(gal_comp[:until,dwarf_inds,0], axis = 1), color = 'tab:blue', linestyle = 'dotted')
plt.plot(z[:until], np.nanmedian(gal_comp[:until,dwarf_inds,3], axis = 1), color = 'tab:orange', linestyle = 'dotted')
plt.plot(z[:until], igm2, color = 'tab:green', linestyle = 'dotted')

# plt.fill_between(z[:until], igm2_low, igm2_high, color = 'tab:green', alpha = 0.2)
# plt.fill_between(z[:until], np.nanpercentile(gal_comp[:until,dwarf_inds,0], 16, axis = 1),\
#                     np.nanpercentile(gal_comp[:until,dwarf_inds,0], 84, axis = 1), color = 'tab:blue', alpha = 0.2)
# plt.fill_between(z[:until], np.nanpercentile(gal_comp[:until,dwarf_inds,3], 16, axis = 1),\
#                     np.nanpercentile(gal_comp[:until,dwarf_inds,3], 84, axis = 1), color = 'tab:orange', alpha = 0.2)

#### MW-like ####

igm3 = np.nanmedian(1 - gal_comp[:until,mw_inds,0] - gal_comp[:until,mw_inds,3], axis = 1)
igm3_high = np.nanpercentile(1 - gal_comp[:until,mw_inds,0] - gal_comp[:until,mw_inds,3], 84, axis = 1)
igm3_low = np.nanpercentile(1 - gal_comp[:until,mw_inds,0] - gal_comp[:until,mw_inds,3], 16, axis = 1)
plt.plot(z[:until], np.nanmedian(gal_comp[:until,mw_inds,0], axis = 1), color = 'tab:blue', linestyle = 'dashdot')
plt.plot(z[:until], np.nanmedian(gal_comp[:until,mw_inds,3], axis = 1), color = 'tab:orange', linestyle = 'dashdot')
plt.plot(z[:until], igm3, color = 'tab:green', linestyle = 'dashdot')

# plt.fill_between(z[:until], igm3_low, igm3_high, color = 'tab:green', alpha = 0.2)
# plt.fill_between(z[:until], np.nanpercentile(gal_comp[:until,mw_inds,0], 16, axis = 1),\
#                     np.nanpercentile(gal_comp[:until,mw_inds,0], 84, axis = 1), color = 'tab:blue', alpha = 0.2)
# plt.fill_between(z[:until], np.nanpercentile(gal_comp[:until,mw_inds,3], 16, axis = 1),\
#                     np.nanpercentile(gal_comp[:until,mw_inds,3], 84, axis = 1), color = 'tab:orange', alpha = 0.2)

#### groups ####

igm4 = np.nanmedian(1 - gal_comp[:until,group_inds,0] - gal_comp[:until,group_inds,3], axis = 1)
igm4_high = np.nanpercentile(1 - gal_comp[:until,group_inds,0] - gal_comp[:until,group_inds,3], 84, axis = 1)
igm4_low = np.nanpercentile(1 - gal_comp[:until,group_inds,0] - gal_comp[:until,group_inds,3], 16, axis = 1)
plt.plot(z[:until], np.nanmedian(gal_comp[:until,group_inds,0], axis = 1), color = 'tab:blue', linestyle = 'dashed')
plt.plot(z[:until], np.nanmedian(gal_comp[:until,group_inds,3], axis = 1), color = 'tab:orange', linestyle = 'dashed')
plt.plot(z[:until], igm4, color = 'tab:green', linestyle = 'dashed')

# plt.fill_between(z[:until], igm4_low, igm4_high, color = 'tab:green', alpha = 0.2)
# plt.fill_between(z[:until], np.nanpercentile(gal_comp[:until,group_inds,0], 16, axis = 1),\
#                     np.nanpercentile(gal_comp[:until,group_inds,0], 84, axis = 1), color = 'tab:blue', alpha = 0.2)
# plt.fill_between(z[:until], np.nanpercentile(gal_comp[:until,group_inds,3], 16, axis = 1),\
#                     np.nanpercentile(gal_comp[:until,group_inds,3], 84, axis = 1), color = 'tab:orange', alpha = 0.2)

solid = mlines.Line2D([], [], color='dimgray', linestyle = 'solid', label='all galaxies')
dot = mlines.Line2D([], [], color='dimgray', linestyle = 'dotted', label='dwarfs')
dashdot = mlines.Line2D([], [], color='dimgray', linestyle = 'dashdot', label='MW-like')
dashed = mlines.Line2D([], [], color='dimgray', linestyle = 'dashed', label='groups')
MP = mpatches.Patch(color='tab:blue', label='main progenitor')
OTHER = mpatches.Patch(color='tab:orange', label='other galaxies')
IGM = mpatches.Patch(color='tab:green', label='IGM')


leg1 = plt.legend(handles = [solid, dashed, dashdot, dot], loc = 'center left')
leg2 = plt.legend(handles = [MP, OTHER, IGM], loc='center right')
plt.gca().add_artist(leg1)


plt.ylabel(r'$M_{\ast,\, \rm insitu}\, (z)\,/\,M_{\ast,\,\rm insitu}\,(z=0)$')
plt.xlabel('redshift')

plt.xlim(0, 6)
plt.tight_layout()
plt.savefig(dirname + f'/tracer_fraction_TNG50-{run}_mass_bins.pdf',format='pdf')