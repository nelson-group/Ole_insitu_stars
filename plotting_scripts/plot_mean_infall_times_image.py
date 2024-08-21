import illustris_python as il
import matplotlib.pyplot as plt
import numpy as np
import h5py

import sys
from os.path import isfile, isdir

plt.style.use('fancy_plots2.mplstyle')


#---- setup ----#

run = int(sys.argv[1])
basePath = f'/virgotng/universe/IllustrisTNG/TNG50-{run}/output'
h_const = il.groupcat.loadHeader(basePath,99)['HubbleParam']
boxSize = il.groupcat.loadHeader(basePath,99)['BoxSize'] / h_const
start_snap = 99

# set the maximum of the radial profiles (determined by the data files you saved, change there accordingly if you want to change here)
num_shmr = 7

# special subhalos (group, MW, dwarf)
sub_ids = np.array([167392, 552879, 706460])

# specify path to your output directory
dirname = 'pics/mean_infall_times'
assert isdir(dirname), 'Output directory does not exist!'

#---- load data ----#

assert isfile('/vera/ptmp/gc/olwitt/auxCats/' + basePath[32:39] + '/accretion_channels_insitu.hdf5'),\
    'Accretion channel file does not exist!'
f = h5py.File('/vera/ptmp/gc/olwitt/auxCats/' + basePath[32:39] + '/accretion_channels_insitu.hdf5','r')
dwarf_ids = f['subhalo_ids/dwarfs'][:]
mw_ids = f['subhalo_ids/mws'][:]
group_ids = f['subhalo_ids/groups'][:]
subhaloFlag = f['subhaloFlag'][:]
f.close()

dwarf_inds = dwarf_ids[np.nonzero(np.isin(dwarf_ids, np.nonzero(subhaloFlag)[0]))[0]]
mw_inds = mw_ids[np.nonzero(np.isin(mw_ids, np.nonzero(subhaloFlag)[0]))[0]]
group_inds = group_ids[np.nonzero(np.isin(group_ids, np.nonzero(subhaloFlag)[0]))[0]]

type_names = np.array(['groups','MW-like','dwarfs'])

fig, ax = plt.subplots(2,2,figsize=(16,16))
ax = ax.flatten()

assert isfile('files/' + basePath[32:39] + f'/mean_infall_times/mean_infall_times_all_subs_{num_shmr}shmr.hdf5'),\
    'Mean infall time file does not exist!'

for i in range(ax.shape[0] - 1):
    f = h5py.File('files/' + basePath[32:39] +\
                  f'/mean_infall_times/mean_infall_times_all_subs_{num_shmr}shmr.hdf5','r')
    star_pos_insitu = f[f'special_sub_{sub_ids[i]}/tracer_pos'][:,:]
    time_means = f[f'special_sub_{sub_ids[i]}/time_means'][:]
    profiles = f['mean_infall_time_profiles'][:,0,:]
    special_profiles = profiles[sub_ids,:]
    dist_bins = f['distance_bins'][:]
    f.close()
    
    print('sub_id: ',sub_ids[i])
    print('star_pos_insitu: ',star_pos_insitu[0,:])
    sub = il.groupcat.loadSingle(basePath, 99, subhaloID = sub_ids[i])
    
    sub_pos = sub['SubhaloPos'] / h_const
    sub_hmr = sub['SubhaloHalfmassRadType'][4] / h_const

    pos = star_pos_insitu[:,:] - sub_pos[:]
    pos[np.where(pos > boxSize)] -= boxSize
    pos[np.where(pos < -boxSize)] += boxSize
    x = pos[:,0]
    y = pos[:,1]
    z = pos[:,2]
    print('subhalo_pos: ',sub_pos)
    print('pos: ',pos[0,:])
    
    mask = np.where(np.logical_and(abs(x)<7*sub_hmr,abs(y)<7*sub_hmr))[0]
    
    ax[-1].plot(dist_bins,special_profiles[i],label = type_names[i])
    # ax[3].fill_between(dist_bins,y16,y84,alpha = 0.3)
    
    #with mpl.rc_context({'figure.figsize' : (4.5,4.5)}):
    circle = plt.Circle((0,0),2*sub_hmr,color = 'white',fill=False,linewidth=0.5)
    ax[i].add_patch(circle)
    im = ax[i].hexbin(x[mask],y[mask],time_means[mask], reduce_C_function = np.nanmedian, \
                             cmap = 'plasma',vmin = 0, vmax = 4, gridsize = 27)
    ax[i].set_xlabel('x [kpc]')
    ax[i].set_ylabel('y [kpc]')
    ax[i].set_aspect('equal')
    ax[i].set_xlim(-5*sub_hmr,5*sub_hmr)
    ax[i].set_ylim(-5*sub_hmr,5*sub_hmr)
    ax[i].text(0.1,0.9,'subhalo ' + str(sub_ids[i]), bbox=\
             dict(boxstyle="round",ec='lightgray',fc='white',alpha = 0.8),transform=ax[i].transAxes,size = 20)

ax[-1].plot(dist_bins,np.nanmedian(profiles[group_inds,:],axis=0), color = 'tab:blue',linestyle = 'dashed')
ax[-1].plot(dist_bins,np.nanmedian(profiles[mw_inds,:],axis=0), color = 'tab:orange',linestyle = 'dashed')
ax[-1].plot(dist_bins,np.nanmedian(profiles[dwarf_inds,:],axis=0), color = 'tab:green',linestyle = 'dashed') 

ax[-1].set_aspect(1.4)
ax[-1].legend()
ax[-1].set_xlabel(r'radial distance [$\rm{R}_{0.5, \ast}$]')
ax[-1].set_ylabel(r'$z_{\rm{infall}}$')
ax[-1].set_ylim(0.5,3.5)
ax[-1].tick_params(axis = 'y', which = 'minor', left = False, right = False)

cb = fig.colorbar(im,label = 'median infall redshift', ax = ax[-1], location = 'top')
cb.minorticks_off()

plt.subplots_adjust(wspace = 0.25)

plt.tight_layout()
plt.savefig(dirname + f'/image_mean_infall_times_{basePath[35:39]}.pdf', format = 'pdf')