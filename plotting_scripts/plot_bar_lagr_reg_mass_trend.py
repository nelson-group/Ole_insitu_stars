import illustris_python as il
import matplotlib.pyplot as plt
import numpy as np
import h5py

import funcs
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

import sys
from os.path import isfile, isdir

plt.style.use('fancy_plots2.mplstyle')

run = int(sys.argv[1])

# specify output directory
dirname = 'pics/lagrangian_regions'
assert isdir(dirname), "Output directory does not exist!"

basePath=f'/virgotng/universe/IllustrisTNG/TNG50-{run}/output'
a = il.groupcat.loadHeader(basePath,0)['Time']
h_const = il.groupcat.loadHeader(basePath, 99)['HubbleParam']

# a = 1

# file data is given in code units, i.e. ckpc/h

assert isfile(f'/vera/ptmp/gc/olwitt/auxCats/TNG50-{run}/baryonic_lagrangian_regions_insitu_test.hdf5'), "In-situ file does not exist!"
f = h5py.File(f'/vera/ptmp/gc/olwitt/auxCats/TNG50-{run}/baryonic_lagrangian_regions_insitu_test.hdf5','r')
subhaloFlag = f['subhaloFlag'][:]
sub_medians = f['subhalo_median_distances'][:,0,0] * a / h_const
sub_99 = f['subhalo_99_percentile_distances'][:,0,0] * a / h_const
sub_95 = f['subhalo_95_percentile_distances'][:,0,0] * a / h_const
sub_90 = f['subhalo_90_percentile_distances'][:,0,0] * a / h_const
radii = f['convex_hull/radii'][:] * a / h_const
f.close()

assert isfile(f'/vera/ptmp/gc/olwitt/auxCats/TNG50-{run}/baryonic_lagrangian_regions_exsitu_test.hdf5'), "Ex-situ file does not exist!"
f = h5py.File(f'/vera/ptmp/gc/olwitt/auxCats/TNG50-{run}/baryonic_lagrangian_regions_exsitu_test.hdf5','r')
subhaloFlag_ex = f['subhaloFlag'][:]
sub_medians_ex = f['subhalo_median_distances'][:] * a / h_const
sub_99_ex = f['subhalo_99_percentile_distances'][:] * a / h_const
sub_95_ex = f['subhalo_95_percentile_distances'][:] * a / h_const
sub_90_ex = f['subhalo_90_percentile_distances'][:] * a / h_const
radii_ex = f['convex_hull/radii'][:] * a / h_const
f.close()

# assert isfile(f'/vera/ptmp/gc/olwitt/auxCats/TNG50-{run}/baryonic_lagrangian_regions_dm_test.hdf5'), "DM file does not exist!"
# f = h5py.File(f'/vera/ptmp/gc/olwitt/auxCats/TNG50-{run}/baryonic_lagrangian_regions_dm_test.hdf5','r')
# subhaloFlag_dm = f['subhaloFlag'][:]
# sub_medians_dm = f['subhalo_median_distances'][:] * a / h_const
# sub_max_dm = f['subhalo_maximum_distances'][:] * a / h_const
# sub_99_dm = f['subhalo_99_percentile_distances'][:,0,0] * a / h_const
# radii_dm = f['convex_hull/radii'][:] * a / h_const
# f.close()

raw_stellar_masses = il.groupcat.loadSubhalos(basePath, 99, fields = ['SubhaloMassType'])[:,4] * 1e10 / h_const

#insitu
numBins = 20
mask = np.where(np.logical_and(subhaloFlag == 1, sub_medians[:] < 10000))[0]
stellar_masses = np.log10(raw_stellar_masses[mask])
mass_bins, sub_median_bins, sub_med_low, sub_med_high = funcs.binData_med(stellar_masses, sub_medians, numBins)
_, radii_bins, rad_low, rad_high = funcs.binData_med(stellar_masses, radii, numBins)
# _, sub_99_bins, sub_99_low, sub_99_high = funcs.binData_med(stellar_masses, sub_99, numBins)
_, sub_95_bins, sub_95_low, sub_95_high = funcs.binData_med(stellar_masses, sub_95, numBins)
# _, sub_90_bins, sub_90_low, sub_90_high = funcs.binData_med(stellar_masses, sub_90, numBins)


# exsitu
mask = np.where(np.logical_and(subhaloFlag_ex == 1, sub_medians_ex[:] < 10000))[0]
stellar_masses = np.log10(raw_stellar_masses[mask])
mass_bins_ex, sub_median_bins_ex, sub_med_low_ex, sub_med_high_ex = funcs.binData_med(stellar_masses, sub_medians_ex, numBins)
_, radii_bins_ex, rad_low_ex, rad_high_ex = funcs.binData_med(stellar_masses, radii_ex, numBins)
# _, sub_99_bins_ex, sub_99_low_ex, sub_99_high_ex = funcs.binData_med(stellar_masses, sub_99_ex, numBins)
_, sub_95_bins_ex, sub_95_low_ex, sub_95_high_ex = funcs.binData_med(stellar_masses, sub_95_ex, numBins)
# _, sub_90_bins_ex, sub_90_low_ex, sub_90_high_ex = funcs.binData_med(stellar_masses, sub_90_ex, numBins)


# # dm
# mask = np.where(np.logical_and(subhaloFlag_dm == 1, sub_medians_dm[:] < 100000))[0]
# stellar_masses = np.log10(raw_stellar_masses[mask])
# mass_bins_dm, sub_median_bins_dm, sub_med_low_dm, sub_med_high_dm = funcs.binData_med(stellar_masses, sub_medians_dm, numBins)
# _, radii_bins_dm, rad_low_dm, rad_high_dm = funcs.binData_med(stellar_masses, radii_dm, numBins)
# _, sub_99_bins_dm, sub_99_low_dm, sub_99_high_dm = funcs.binData_med(stellar_masses, sub_99_dm, numBins)
# _, sub_95_bins_dm, sub_95_low_dm, sub_95_high_dm = funcs.binData_med(stellar_masses, sub_95_dm, numBins)
# -, sub_90_bins_dm, sub_90_low_dm, sub_90_high_dm = funcs.binData_med(stellar_masses, sub_90_dm, numBins)

fig,ax = plt.subplots(1,1,figsize=(8,6))
plt.plot(mass_bins, sub_median_bins, color = 'tab:blue')
plt.plot(mass_bins_ex, sub_median_bins_ex, color = 'tab:blue', linestyle = 'dashed')

plt.plot(mass_bins, radii_bins, color = 'tab:green')
plt.plot(mass_bins_ex, radii_bins_ex, color = 'tab:green', linestyle = 'dashed')

plt.plot(mass_bins, sub_95_bins, color = 'tab:orange', linestyle = 'solid')
plt.plot(mass_bins_ex, sub_95_bins_ex, color = 'tab:orange', linestyle = 'dashed')

solid = mlines.Line2D([], [], color = 'gray', linestyle = 'solid',\
                      label = f'in-situ')
dotted = mlines.Line2D([], [], color = 'gray', linestyle = 'dashed',\
                       label = f'ex-situ')
dashed = mlines.Line2D([], [], color = 'gray', linestyle = 'dotted',\
                       label = f'DM')

blue = mpatches.Patch(color = 'tab:blue', label = 'median distance')
green = mpatches.Patch(color = 'tab:green', label = 'convex hull radius')
orange = mpatches.Patch(color = 'tab:orange', label = r'95$^{\rm{th}}$ percentile distance')

legend1 = plt.legend(handles=[blue,orange, green], loc = 'best')

# legend2 = plt.legend(handles = [solid,dotted,dashed], loc = 'upper right')
# plt.gca().add_artist(legend1)

plt.yscale('log')
plt.xlabel(r'stellar mass [$\log\, \rm M_\odot$]')
plt.ylabel('spatial extent [pkpc]')

plt.xlim(9,12)
plt.ylim(10,5000)

ax.set_xticks([9,10,11,12])
ax.set_xticklabels([9,10,11,12])

# plt.hlines(1230,8,12,linestyle='dotted',color='gray', label = '0.5 boxSize)
# plt.hlines(2130,8,12,linestyle='dotted',color='black', label = 'sqrt(3)/2 boxSize)

plt.text(11, 15,'z = 20', size = 20, bbox=\
             dict(boxstyle="round",ec='black',fc='lightgray',alpha = 0.5))

plt.tight_layout()
plt.savefig(dirname + f'/median_distance_vs_mass_50-{run}_small.pdf',format='pdf')