import illustris_python as il
import matplotlib.pyplot as plt
import numpy as np
import h5py
from numba import jit, njit
import funcs
from os.path import isfile, isdir
import os

import sys

import matplotlib.lines as mlines
import matplotlib.patches as mpatches

plt.style.use('fancy_plots2.mplstyle')

#compute fractions for ALL galaxies. however, only a small amount of centrals was really used -> retrieve that information
# @njit
def sub_fracs(subhaloFlag, offsets, location_at_snap99, star_formation_distances):
    num_subs = offsets.shape[0] - 1
    form_out_loc_out = np.full(num_subs, 0, dtype = np.float32)
    form_out_loc_in = np.full(num_subs, 0, dtype = np.float32)
    form_in_loc_out = np.full(num_subs, 0, dtype = np.float32)
    form_in_loc_in = np.full(num_subs, 0, dtype = np.float32)
    medsitu = np.full(num_subs, 0, dtype = np.float32)
    insitu = np.full(num_subs, 0, dtype = np.float32)
    
    for i in range(num_subs):
        if subhaloFlag[i] == 0:
            continue
        indices_of_sub = np.arange(offsets[i],offsets[i+1])
        if indices_of_sub.shape[0] > 0:

            #---
            in_in = np.where(np.logical_and(star_formation_distances[indices_of_sub] < 2, location_at_snap99[indices_of_sub] == 1))[0]
            form_in_loc_in[i] = in_in.shape[0] / indices_of_sub.shape[0]
            #---
            in_out = np.where(np.logical_and(star_formation_distances[indices_of_sub] < 2, location_at_snap99[indices_of_sub] == 0))[0]
            form_in_loc_out[i] = in_out.shape[0] / indices_of_sub.shape[0]
            #---
            out_in = np.where(np.logical_and(star_formation_distances[indices_of_sub] >= 2, location_at_snap99[indices_of_sub] == 1))[0]
            form_out_loc_in[i] = out_in.shape[0] / indices_of_sub.shape[0]
            #---
            out_out = np.where(np.logical_and(star_formation_distances[indices_of_sub] >= 2, location_at_snap99[indices_of_sub] == 0))[0]
            form_out_loc_out[i] = out_out.shape[0] / indices_of_sub.shape[0]

            assert in_in.shape[0] + in_out.shape[0] + out_in.shape[0] + out_out.shape[0] == indices_of_sub.shape[0]

            # correct medsitu definition
            medsitu_tmp = np.where(star_formation_distances[indices_of_sub] > 2)[0]
            medsitu[i] = medsitu_tmp.shape[0] / indices_of_sub.shape[0]        

            insitu_tmp = np.where(star_formation_distances[indices_of_sub] <= 2)[0]
            insitu[i] = insitu_tmp.shape[0] / indices_of_sub.shape[0]

    return form_in_loc_in, form_in_loc_out, form_out_loc_in, form_out_loc_out, medsitu, insitu

run = int(sys.argv[1])
basePath=f'/virgotng/universe/IllustrisTNG/TNG50-{run}/output'
stype = 'insitu'

# set number of stellar mass bins
num_bins = 15

# specify path to your directory to save the output plot
dirname = 'pics/med-situ'
assert isdir(dirname), 'Directory not found!'

# load star formation distances

assert isfile(f'/vera/ptmp/gc/olwitt/{stype}/TNG50-{run}/star_formation_distances.hdf5'), 'Star formation distance file not found!'
f = h5py.File('/vera/ptmp/gc/olwitt/' + stype + f'/TNG50-{run}/star_formation_distances.hdf5','r')
star_formation_distances = f['star_formation_distances'][:]
f.close()

file = '/vera/ptmp/gc/olwitt/' + stype + f'/TNG50-{run}/distance_cats/distance_cats_99.hdf5'
assert isfile(file), 'Distance catalog file not found!'
f = h5py.File(file,'r')
subhaloFlag_99 = f['subhaloFlag'][:]
inside_radius = f['tracers_inside_radius'][:]
final_offsets = f['subhalo_offsets'][:]
f.close()

location_status = np.full(inside_radius.shape[0], -1, dtype = np.byte)
location_status[np.where(inside_radius > 99)[0]] = 1
del inside_radius

# which subs were used for analysis (at snapshot 99)
subs = np.nonzero(subhaloFlag_99)[0]

# execute function
form_in_loc_in, form_in_loc_out, form_out_loc_in, form_out_loc_out, medsitu, insitu =\
      sub_fracs(subhaloFlag_99, final_offsets, location_status, star_formation_distances)

# load stellar masses and convert to log10
h_const = il.groupcat.loadHeader(basePath, 99)['HubbleParam']
stellar_masses = il.groupcat.loadSubhalos(basePath, 99, fields = ['SubhaloMassType'])[:,4] * 1e10 / h_const
stellar_masses = funcs.log10_mod(stellar_masses[subs])

# bin fractions regarding stellar mass
mass_bins, form_in_loc_in_bins,_,_ = funcs.binData_med(stellar_masses, form_in_loc_in[subs], num_bins)
_, form_in_loc_out_bins,_,_ = funcs.binData_med(stellar_masses, form_in_loc_out[subs], num_bins)
_, form_out_loc_in_bins,_,_ = funcs.binData_med(stellar_masses, form_out_loc_in[subs], num_bins)
_, form_out_loc_out_bins,_,_ = funcs.binData_med(stellar_masses, form_out_loc_out[subs], num_bins)
_, med_situ_bins,_,_ = funcs.binData_med(stellar_masses, medsitu[subs], num_bins)
_, in_situ_bins,_,_ = funcs.binData_med(stellar_masses, insitu[subs], num_bins)


# plot results
fig, ax = plt.subplots(1,1, figsize = (8,6))

ax.plot(mass_bins, form_in_loc_in_bins, color = 'tab:blue', linestyle = 'solid')
ax.plot(mass_bins, form_out_loc_in_bins, color = 'tab:blue', linestyle = 'dashed')
ax.plot(mass_bins, form_in_loc_out_bins, color = 'tab:orange', linestyle = 'solid')
ax.plot(mass_bins, form_out_loc_out_bins, color = 'tab:orange', linestyle = 'dashed')

plt.plot(mass_bins, in_situ_bins, color = 'black', linestyle = 'solid')
plt.plot(mass_bins, med_situ_bins, color = 'black', linestyle = 'dashed')

ax.set_xlabel(r'stellar mass [$\log\;\rm M_\odot$]')
ax.set_ylabel('in-situ stellar mass fraction')

loc_in = mpatches.Patch(color='tab:blue', label=r'inside $2\,R_{\rm SF, 1/2}$ at $z=0$')
loc_out = mpatches.Patch(color='tab:orange', label=r'outside $2\,R_{\rm SF, 1/2}$ at $z=0$')
form_in = mlines.Line2D([], [], color='tab:gray', linestyle = 'solid', label=r'inside 2 $R_{\rm SF,1/2}$ at formation')
form_out = mlines.Line2D([], [], color='tab:gray', linestyle = 'dashed', label=r'outside 2 $R_{\rm SF,1/2}$ at formation')

legend1 = plt.legend(handles = [form_in, form_out, loc_in, loc_out], bbox_to_anchor=(0.0, 0.95), loc = 'upper left', fontsize = 19)

ax.set_xlim(9,12)
ax.set_ylim(0.005,1.02)
ax.set_yscale('log')


ax.set_xticks(np.array([9,10,11,12]))
ax.set_xticklabels(['9','10','11','12'])

fig.tight_layout()

plt.savefig(dirname + f'/true_false_insitu_50-{run}_2HMR_Sfr_Gas.pdf', format = 'pdf')