import matplotlib.pyplot as plt
import numpy as np
import h5py
import illustrisFuncs as iF

import matplotlib.lines as mlines
import sys
from os.path import isfile, isdir
plt.style.use('fancy_plots2.mplstyle')

run = int(sys.argv[1])
basePath = f'/virgotng/universe/IllustrisTNG/TNG50-{run}/output'

# specify whether to plot profiles binned w.r.t. r_vir or shmr
what_to_plot = 'r_vir'

# specify xscale
xscale = 'log'

# specify output directory
dirname = 'files/radial_profiles'
assert isdir(dirname), 'output directory does not exist'

# corresponds to the number of bins entered in lagrangian_regions_times.py
numBins = 201
num_r_vir = 15
# num_hmr = int(num_r_vir * 40 / 3) #200shmr or 15 r_vir are good numbers
# dist_bins_hmr = np.linspace(0,num_hmr,numBins)

dist_bins = np.linspace(0,num_r_vir,numBins)

# specify snapshots which are plotted
snaps = np.array([99,84,67,50,33,25])
z = iF.give_z_array(basePath)
z_snaps = np.flip(z)[snaps]

fig, ax = plt.subplots(2,3,figsize = (16,12))
ax = ax.flatten()
loc = dist_bins[int(dist_bins.size/2)] if xscale == 'linear' else dist_bins[int(dist_bins.size/10)]
style = ['solid','dotted','dashed']

for i in range(snaps.size):
    file = f'/vera/ptmp/gc/olwitt/insitu/TNG50-{run}/lagrangian_regions/lagrangian_regions_w_profiles_{snaps[i]}_cumulative.hdf5'
    assert isfile(file), ' Profile file does not exist!'
    f = h5py.File(file,'r')
    profiles = f['cumulative_radial_profiles_'+what_to_plot][:,:,:]
    # profiles_situ = f['cumulative_radial_profiles_situ_'+what_to_plot][:,:,:]
    subhaloFlag = f['subhaloFlag'][:]
    hmr = f['lagrangian_regions_'+what_to_plot][:,:,:] 
    
    dwarf_ids = f['mass_bin_sub_ids/dwarfs'][:]
    dwarf_ids = dwarf_ids[np.nonzero(subhaloFlag[dwarf_ids])]
    mw_ids = f['mass_bin_sub_ids/mws'][:]
    mw_ids = mw_ids[np.nonzero(subhaloFlag[mw_ids])]
    group_ids = f['mass_bin_sub_ids/groups'][:]
    group_ids = group_ids[np.nonzero(subhaloFlag[group_ids])]

    f.close()
    
    ptype = 0 #0 for regular, 1 for igm, 2 for satellites
    situ = 0 #0 for all tracers, 1 for insitu, 2 for med-situS

    ax[0].plot(dist_bins, np.nanmedian(profiles[dwarf_ids,ptype,:], axis=0),\
                 label = f'z = {z_snaps[i]:.1f}')
    ax[0].scatter(np.nanmedian(hmr[dwarf_ids,ptype]),0.5)

    if i == 4:
        ax[0].fill_between(dist_bins, np.nanpercentile(profiles[dwarf_ids,ptype,:], 16, axis=0),\
                            np.nanpercentile(profiles[dwarf_ids,ptype,:], 84, axis=0), alpha = 0.5, color = f'C{i}')

    ax[1].plot(dist_bins, np.nanmedian(profiles[mw_ids,ptype,:], axis=0),\
                 label = f'z = {z_snaps[i]:.1f}')
    ax[1].scatter(np.nanmedian(hmr[mw_ids,ptype]),0.5)

    if i == 4:
        ax[1].fill_between(dist_bins, np.nanpercentile(profiles[mw_ids,ptype,:], 16, axis=0),\
                            np.nanpercentile(profiles[mw_ids,ptype,:], 84, axis=0), alpha = 0.5, color = f'C{i}')
    
    ax[2].plot(dist_bins, np.nanmedian(profiles[group_ids,ptype,:], axis=0),\
                 label = f'z = {z_snaps[i]:.1f}')
    ax[2].scatter(np.nanmedian(hmr[group_ids,ptype]),0.5)

    if i == 4:
        ax[2].fill_between(dist_bins, np.nanpercentile(profiles[group_ids,ptype,:], 16, axis=0),\
                            np.nanpercentile(profiles[group_ids,ptype,:], 84, axis=0), alpha = 0.5, color = f'C{i}')
    
    #---- second row -----#
    
    if snaps[i] in [50,67,33]:
        for ptype in range(3):
            ax[3].plot(dist_bins, np.nanmedian(profiles[dwarf_ids,ptype,:], axis=0),\
                              color = 'C'+str(i),linestyle = style[ptype])
            ax[4].plot(dist_bins, np.nanmedian(profiles[mw_ids,ptype,:], axis=0),\
                              color = 'C'+str(i),linestyle = style[ptype])
            ax[5].plot(dist_bins, np.nanmedian(profiles[group_ids,ptype,:], axis=0),\
                              color = 'C'+str(i),linestyle = style[ptype])

            if snaps[i] == 33:
                ax[3].fill_between(dist_bins, np.nanpercentile(profiles[dwarf_ids,ptype,:], 16, axis=0),\
                                    np.nanpercentile(profiles[dwarf_ids,ptype,:], 84, axis=0), alpha = 0.5)
                ax[4].fill_between(dist_bins, np.nanpercentile(profiles[mw_ids,ptype,:], 16, axis=0),\
                                    np.nanpercentile(profiles[mw_ids,ptype,:], 84, axis=0), alpha = 0.5)
                ax[5].fill_between(dist_bins, np.nanpercentile(profiles[group_ids,ptype,:], 16, axis=0),\
                                    np.nanpercentile(profiles[group_ids,ptype,:], 84, axis=0), alpha = 0.5)

            
        # for situ in range(3):
        #     ax[6].plot(dist_bins, np.nanmedian(profiles_situ[dwarf_ids,situ,:], axis=0),\
        #                       color = 'C'+str(i),linestyle = style[situ])
        #     ax[7].plot(dist_bins, np.nanmedian(profiles_situ[mw_ids,situ,:], axis=0),\
        #                       color = 'C'+str(i),linestyle = style[situ])
        #     ax[8].plot(dist_bins, np.nanmedian(profiles_situ[group_ids,situ,:], axis=0),\
        #                       color = 'C'+str(i),linestyle = style[situ])
    print(f'snap {snaps[i]} done', flush = True)
    
# ax[0].text(loc,0.1,'dwarfs', size = 20, bbox=\
#              dict(boxstyle="round",ec='black',fc='lightgray',alpha = 0.5))
# ax[1].text(loc,0.1,'MW-like', size = 20, bbox=\
#              dict(boxstyle="round",ec='black',fc='lightgray',alpha = 0.5))
# ax[2].text(loc,0.3,'groups', size = 20, bbox=\
#              dict(boxstyle="round",ec='black',fc='lightgray',alpha = 0.5))

ax[0].set_title('dwarfs',size = 24)
ax[1].set_title('MW-like',size = 24)
ax[2].set_title('groups',size = 24)

ax[2].legend(ncol = 2,loc='lower right', fontsize = 20)
        
solid = mlines.Line2D([], [], color = 'gray', linestyle = 'solid', label = 'all tracers')
dotted = mlines.Line2D([], [], color = 'gray', linestyle = 'dotted', label = 'from IGM')
dashed = mlines.Line2D([], [], color = 'gray', linestyle = 'dashed', label = 'from mergers')

all_tracers = mlines.Line2D([], [], color = 'gray', linestyle = 'solid', label = 'all tracers')
insitu = mlines.Line2D([], [], color = 'gray', linestyle = 'dotted', label = 'in-situ')
medsitu = mlines.Line2D([], [], color = 'gray', linestyle = 'dashed', label = 'med-situ')

ax[5].legend(handles = [solid,dotted,dashed])
# ax[8].legend(handles = [all_tracers, insitu, medsitu])

ax[0].set_ylabel(r'mass fraction $\rm{M}(\!<r)/\rm{M}_{\rm{tot}}$')
ax[3].set_ylabel(r'mass fraction $\rm{M}(\!<r)/\rm{M}_{\rm{tot}}$')
# ax[6].set_ylabel(r'mass fraction $\rm{M}(\!<r)/\rm{M}_{\rm{tot}}$')
ax[-3].set_xlabel(r'radial distance [$\rm{R}_{\rm{200c}}$]')
ax[-2].set_xlabel(r'radial distance [$\rm{R}_{\rm{200c}}$]')
ax[-1].set_xlabel(r'radial distance [$\rm{R}_{\rm{200c}}$]')

    
low = min(dist_bins) + dist_bins[1] if xscale == 'log' else min(dist_bins)
top = max(dist_bins)
print(low,top)
for i in range(len(ax)):
    ax[i].set_xscale(xscale)
    ax[i].set_xticks([0.1,1.0,10.0])
    ax[i].set_xticklabels([0.1,1.0,10.0])
    ax[i].set_xlim(low,top)
    ax[i].minorticks_on()
    ax[i].set_ylim(-0.04,1.04)
ax[0].tick_params(labelbottom = False)
# ax[3].tick_params(labelbottom = False)
ax[1].tick_params(labelleft = False, labelbottom = False)
ax[2].tick_params(labelleft = False, labelbottom = False)
ax[4].tick_params(labelleft = False)
ax[5].tick_params(labelleft = False)
# ax[7].tick_params(labelleft = False)
# ax[8].tick_params(labelleft = False)
plt.tight_layout()
plt.subplots_adjust(wspace = 0.05, hspace = 0.05)


plt.savefig(dirname + f'/rad_dist_profile_TNG50-{run}_{what_to_plot}.pdf',format='pdf')