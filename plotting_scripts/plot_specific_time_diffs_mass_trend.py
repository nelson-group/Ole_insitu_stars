import time
import illustris_python as il
import matplotlib.pyplot as plt
import numpy as np
import h5py
import numba as nb
from numba import jit, njit
import tracerFuncs as tF
import locatingFuncs as lF
import illustrisFuncs as iF
import funcs
import matplotlib as mpl
from matplotlib.patches import Rectangle
from os.path import isfile, isdir
import os
import dm

import sys

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib as mpl

plt.style.use('fancy_plots2.mplstyle')

#compute fractions for ALL galaxies. however, only a small amount of centrals was really used -> retrieve that information
@njit
def sub_diff(offsets, halo_infall_snaps, first_halo_infall_snaps, galaxy_infall_snaps, first_galaxy_infall_snaps, star_formation_snaps, star_formation_distances, a):
    num_subs = offsets.shape[0] - 1
    #works for always outside, in-situ, med-situ
    halo_star_form_diff = np.full((num_subs, 3), np.nan, dtype = np.float32)
    #works for in-situ, med-situ
    halo_gal_diff = np.full((num_subs, 3), np.nan, dtype = np.float32)
    #works for in-situ, med-situ (but negative for med-situ!)
    gal_star_form_diff = np.full((num_subs, 3), np.nan, dtype = np.float32)

    first_last_halo_infall_diff = np.full((num_subs,3), np.nan, dtype = np.float32)
    first_last_galaxy_infall_diff = np.full((num_subs,3), np.nan, dtype = np.float32)
    
    for i in range(num_subs):
        indices_of_sub = np.arange(offsets[i],offsets[i+1])
        if indices_of_sub.shape[0] > 0:

            medsitu = np.where(star_formation_distances[indices_of_sub] > 2)[0]
            insitu = np.where(star_formation_distances[indices_of_sub] <= 2)[0]

            halo_gal_diff_tmp = np.abs(iF.get_time_difference_in_Gyr(a[halo_infall_snaps[indices_of_sub]],\
                                                           a[galaxy_infall_snaps[indices_of_sub]]))
            
            halo_star_form_diff_tmp = np.abs(iF.get_time_difference_in_Gyr(a[halo_infall_snaps[indices_of_sub]],\
                                                           a[star_formation_snaps[indices_of_sub]]))
            gal_star_form_diff_tmp = np.abs(iF.get_time_difference_in_Gyr(a[galaxy_infall_snaps[indices_of_sub]],\
                                                           a[star_formation_snaps[indices_of_sub]]))
            
            first_last_halo_infall_diff_tmp = np.abs(iF.get_time_difference_in_Gyr(a[first_halo_infall_snaps[indices_of_sub]],\
                                                              a[halo_infall_snaps[indices_of_sub]]))
            first_last_galaxy_infall_diff_tmp = np.abs(iF.get_time_difference_in_Gyr(a[first_galaxy_infall_snaps[indices_of_sub]],\
                                                                a[galaxy_infall_snaps[indices_of_sub]]))
            
            halo_gal_diff[i,0] = np.nanmedian(halo_gal_diff_tmp) #all
            halo_star_form_diff[i,0] = np.nanmedian(halo_star_form_diff_tmp) #all
            gal_star_form_diff[i,0] = np.nanmedian(gal_star_form_diff_tmp) #all
            first_last_galaxy_infall_diff[i,0] = np.nanmedian(first_last_galaxy_infall_diff_tmp) #all
            first_last_halo_infall_diff[i,0] = np.nanmedian(first_last_halo_infall_diff_tmp) #all
            
            if insitu.shape[0] > 0:
                halo_gal_diff[i,1] = np.nanmedian(halo_gal_diff_tmp[insitu]) #insitu
                halo_star_form_diff[i,1] = np.nanmedian(halo_star_form_diff_tmp[insitu]) #insitu
                gal_star_form_diff[i,1] = np.nanmedian(gal_star_form_diff_tmp[insitu]) #insitu
                first_last_galaxy_infall_diff[i,1] = np.nanmedian(first_last_galaxy_infall_diff_tmp[insitu]) #insitu
                first_last_halo_infall_diff[i,1] = np.nanmedian(first_last_halo_infall_diff_tmp[insitu]) #insitu
            
            if medsitu.shape[0] > 0:
                halo_gal_diff[i,2] = np.nanmedian(halo_gal_diff_tmp[medsitu]) #medsitu
                halo_star_form_diff[i,2] = np.nanmedian(halo_star_form_diff_tmp[medsitu]) #medsitu
                gal_star_form_diff[i,2] = np.nanmedian(gal_star_form_diff_tmp[medsitu]) #medsitu
                first_last_galaxy_infall_diff[i,2] = np.nanmedian(first_last_galaxy_infall_diff_tmp[medsitu])
                first_last_halo_infall_diff[i,2] = np.nanmedian(first_last_halo_infall_diff_tmp[medsitu])
                
    return halo_star_form_diff, halo_gal_diff, gal_star_form_diff, first_last_halo_infall_diff, first_last_galaxy_infall_diff

#plot difference between halo infall and galaxy infall as well as halo infall and star formation
#as well as galaxy infall and star formation

run = int(sys.argv[1])
basePath = f'/virgotng/universe/IllustrisTNG/TNG50-{run}/output'
stype = 'insitu'
num_bins = 20

#specify path to output directory
dirname = 'pics/med-situ'
assert isdir(dirname), 'Directory does not exist!'

#---- load data ----#

file = '/vera/ptmp/gc/olwitt/' + stype + f'/TNG50-{run}/infall_and_leaving_times.hdf5'
assert isfile(file), 'Infall and leaving times file does not exist!'
f = h5py.File(file,'r')
halo_infall_snaps = f['halo_infall'][:]
galaxy_infall_snaps = f['galaxy_infall'][:]
IGM_leaving = f['leaving_igm'][:]
first_halo_infall_snaps = f['first_halo_infall'][:]
first_galaxy_infall_snaps = f['first_galaxy_infall'][:]
f.close()

assert isfile('/vera/ptmp/gc/olwitt/' + stype + f'/TNG50-{run}/star_formation_snapshots.hdf5'), 'Star formation snapshots file does not exist!'
f = h5py.File('/vera/ptmp/gc/olwitt/' + stype + f'/TNG50-{run}/star_formation_snapshots.hdf5','r')
star_formation_snaps = f['star_formation_snapshot'][:]
f.close()

assert isfile('/vera/ptmp/gc/olwitt/' + stype + f'/TNG50-{run}/star_formation_distances.hdf5'), 'Star formation distances file does not exist!'
f = h5py.File('/vera/ptmp/gc/olwitt/' + stype + f'/TNG50-{run}/star_formation_distances.hdf5','r')
star_formation_distances = f['star_formation_distances'][:]
f.close()

assert isfile('/vera/ptmp/gc/olwitt/' + stype + f'/TNG50-{run}/parent_indices_99.hdf5'), 'Parent indices file does not exist!'
f = h5py.File('/vera/ptmp/gc/olwitt/' + stype + f'/TNG50-{run}/parent_indices_99.hdf5','r')
numTracersInParents = f['snap_99/numTracersInParents'][:]
f.close()
insituStarsInSubOffset = tF.insituStarsInSubOffset(basePath,99)
final_offsets = tF.tracersInSubhalo(insituStarsInSubOffset,numTracersInParents).astype(int)
final_offsets = np.insert(final_offsets,0,0)

z = np.flip(iF.give_z_array(basePath))
a = 1/(1+z)

file = '/vera/ptmp/gc/olwitt/auxCats/' + basePath[32:39] + '/subhaloFlag_' + stype + '.hdf5'
assert isfile(file), 'Subhalo flag file does not exist!'
f = h5py.File(file,'r')
subhaloFlag_99 = f['subhaloFlag'][:]
f.close()

#which subs are used for analysis (at snapshot 99)
subs = np.nonzero(subhaloFlag_99)[0]

h_const = il.groupcat.loadHeader(basePath, 99)['HubbleParam']
stellar_masses = il.groupcat.loadSubhalos(basePath, 99, fields = ['SubhaloMassType'])[:,4] * 1e10 / h_const
stellar_masses = funcs.log10_mod(stellar_masses[subs])

#---- compute differences ----#

halo_star_form_diff, halo_gal_diff, gal_star_form_diff, first_last_halo_infall_diff, first_last_galaxy_infall_diff =\
      sub_diff(final_offsets, halo_infall_snaps, first_halo_infall_snaps, galaxy_infall_snaps, first_galaxy_infall_snaps,\
                star_formation_snaps, star_formation_distances, a)

#---- plot ----#

halo_star_form_diff = halo_star_form_diff[subs,:]
gal_star_form_diff = gal_star_form_diff[subs,:]
halo_gal_diff = halo_gal_diff[subs,:]
first_last_halo_infall_diff = first_last_halo_infall_diff[subs,:]
first_last_galaxy_infall_diff = first_last_galaxy_infall_diff[subs,:]

mass_bins, halo_star_form_diff_bins_all,_,_ = funcs.binData_med(stellar_masses, halo_star_form_diff[:,0], num_bins)
_, halo_star_form_diff_bins_in,_,_ = funcs.binData_med(stellar_masses, halo_star_form_diff[:,1], num_bins)
_, halo_star_form_diff_bins_med,_,_ = funcs.binData_med(stellar_masses, halo_star_form_diff[:,2], num_bins)

_, gal_star_form_diff_bins_in,_,_ = funcs.binData_med(stellar_masses, gal_star_form_diff[:,1], num_bins)
_, gal_star_form_diff_bins_med,_,_ = funcs.binData_med(stellar_masses, gal_star_form_diff[:,2], num_bins)
_, gal_star_form_diff_bins_all,_,_ = funcs.binData_med(stellar_masses, gal_star_form_diff[:,0], num_bins)

_, halo_gal_diff_bins_in,_,_ = funcs.binData_med(stellar_masses, halo_gal_diff[:,1], num_bins)
_, halo_gal_diff_bins_med,_,_ = funcs.binData_med(stellar_masses, halo_gal_diff[:,2], num_bins)
_, halo_gal_diff_bins_all,_,_ = funcs.binData_med(stellar_masses, halo_gal_diff[:,0], num_bins)

_, first_last_halo_infall_diff_bins_in,_,_ = funcs.binData_med(stellar_masses, first_last_halo_infall_diff[:,1], num_bins)
_, first_last_halo_infall_diff_bins_med,_,_ = funcs.binData_med(stellar_masses, first_last_halo_infall_diff[:,2], num_bins)
_, first_last_halo_infall_diff_bins_all,_,_ = funcs.binData_med(stellar_masses, first_last_halo_infall_diff[:,0], num_bins)

_, first_last_galaxy_infall_diff_bins_in,_,_ = funcs.binData_med(stellar_masses, first_last_galaxy_infall_diff[:,1], num_bins)
_, first_last_galaxy_infall_diff_bins_med,_,_ = funcs.binData_med(stellar_masses, first_last_galaxy_infall_diff[:,2], num_bins)
_, first_last_galaxy_infall_diff_bins_all,_,_ = funcs.binData_med(stellar_masses, first_last_galaxy_infall_diff[:,0], num_bins)


fig, ax = plt.subplots(1,1, figsize = (8,6))

plt.plot(mass_bins, halo_star_form_diff_bins_all, color = 'tab:blue', linestyle = 'solid')
# plt.plot(mass_bins, halo_star_form_diff_bins_in, color = 'tab:blue', linestyle = 'dashed')
# plt.plot(mass_bins, halo_star_form_diff_bins_med, color = 'tab:blue', linestyle = 'dotted')
# plt.plot(mass_bins, halo_star_form_diff_bins_out, color = 'tab:blue', linestyle = 'dashdot')

plt.plot(mass_bins, gal_star_form_diff_bins_all, color = 'tab:orange', linestyle = 'solid')
# plt.plot(mass_bins, gal_star_form_diff_bins_in, color = 'tab:orange', linestyle = 'dashed')
# plt.plot(mass_bins, gal_star_form_diff_bins_med, color = 'tab:orange', linestyle = 'dotted')

plt.plot(mass_bins, halo_gal_diff_bins_all, color = 'tab:green', linestyle = 'solid')
# plt.plot(mass_bins, halo_gal_diff_bins_in, color = 'tab:green', linestyle = 'dashed')
# plt.plot(mass_bins, halo_gal_diff_bins_med, color = 'tab:green', linestyle = 'dotted')


# plt.plot(mass_bins, np.sum(np.array([halo_gal_diff_bins_all, \
#                                         gal_star_form_diff_bins_all]), axis = 0), color = 'tab:gray', linestyle = 'solid')
plt.plot(mass_bins, first_last_galaxy_infall_diff_bins_all, color = 'tab:red', linestyle = 'solid')
# plt.plot(mass_bins, first_last_galaxy_infall_diff_bins_in, color = 'tab:red', linestyle = 'dashed')
# plt.plot(mass_bins, first_last_galaxy_infall_diff_bins_med, color = 'tab:red', linestyle = 'dotted')

plt.xlabel(r'stellar mass [$\log\,\rm M_\odot$]')
plt.ylabel('time between events [Gyr]')

halo_star_form = mpatches.Patch(color='tab:blue', label = 'halo infall - star formation')
gal_star_form = mpatches.Patch(color='tab:orange', label = 'galaxy infall - star formation')
halo_gal = mpatches.Patch(color='tab:green', label = 'halo infall - galaxy infall')
first_last_galaxy = mpatches.Patch(color='tab:red', label = 'first - last galaxy infall')

all_stars = mlines.Line2D([], [], color='tab:gray', linestyle = 'solid', label = 'all stars')
in_stars = mlines.Line2D([], [], color='tab:gray', linestyle = 'dashed', label = 'in-situ stars')
med_stars = mlines.Line2D([], [], color='tab:gray', linestyle = 'dotted', label = 'med-situ stars')

legend1 = plt.legend(handles = [halo_star_form, gal_star_form, halo_gal, first_last_galaxy], loc= 'lower left')
# legend2 = plt.legend(handles = [all_stars, in_stars, med_stars], loc= 'lower left')
# plt.gca().add_artist(legend1)

plt.xlim(9,12.0)
plt.yscale('log')
plt.ylim(0.1,10.2)

ax.set_xticks([9,10,11,12])
ax.set_xticklabels(['9','10','11','12'])

plt.tight_layout()
plt.savefig(dirname + f'/specific_time_diffs_50-{run}_small_wo_med-situ.pdf', format = 'pdf')