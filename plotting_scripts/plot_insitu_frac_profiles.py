import illustris_python as il
import matplotlib.pyplot as plt
import numpy as np
import h5py
import numba as nb
from numba import jit
import funcs
import matplotlib.patches as mpatches
from os.path import isfile, isdir

import sys

plt.style.use('fancy_plots2.mplstyle')

@jit(nopython = True)
def ex_in_frac_binning(starType, distance, numBins, max_dist): 
    minVal = 0
    maxVal = max_dist
    
    binWidth = (maxVal - minVal) / numBins
    
    yMed = np.zeros(numBins)
    
    for j in range(numBins):
        relInd = np.where(np.logical_and(distance >= minVal + j*binWidth, distance < minVal + (j+1)*binWidth))[0]
        if(relInd.size>0):
            insitu = np.where(starType[relInd] == 1)[0].shape[0]
            exsitu = np.where(starType[relInd] == 0)[0].shape[0]
            yMed[j] = insitu/(insitu + exsitu)# if insitu > 0 else np.nan
            
    return yMed

@jit(nopython = True, parallel = True)
def compute_profiles(sub_ids, num_bins, offsets, numPart_stars, sub_pos, starType, star_pos, boxSize,\
                                    group_R_Crit200, num_gmcrit200):
    sub_profiles = np.zeros((sub_ids.shape[0],num_bins))
    
    for i in nb.prange(sub_ids.shape[0]):
        offset = offsets[i]
        num_stars = numPart_stars[i]
        subhalo_pos = sub_pos[i,:]
        indcs = np.arange(offset,offset + num_stars)
        starType_sub = starType[indcs]        
        distances = funcs.dist_vector_nb(subhalo_pos,star_pos[indcs],boxSize) #/h_const
    
        max_dist = num_gmcrit200 * group_R_Crit200[i]#subs['SubhaloHalfmassRadType'][sub_ids[i],4]
        sub_profiles[i,:] = ex_in_frac_binning(starType_sub, distances, num_bins, max_dist)
    return sub_profiles

def exsitu_radial_profile(basePath, snap, sub_ids, group_R_Crit200, num_bins = 10, num_gmcrit200 = 2,\
                          star_pos = 0, subs = 0):
    """Computes (average) radial profile of given subhalos at specific snapshot up until max_dist times the
    virial radius."""
    boxSize = il.groupcat.loadHeader(basePath,snap)['BoxSize']
    
    f = h5py.File(basePath[:-6]+'postprocessing/StellarAssembly/stars_' + str(snap).zfill(3) + '.hdf5','r')
    starType = f['InSitu'][:]
    f.close()
    
    g = h5py.File(basePath[:-6] + 'postprocessing/offsets/offsets_' + str(snap).zfill(3) + '.hdf5','r')
    offset = g['Subhalo/SnapByType'][sub_ids,4]
    g.close()
    
    numPart_stars = subs['SubhaloLenType'][sub_ids,4]
    sub_pos = subs['SubhaloPos'][sub_ids,:]
    
    sub_profiles = compute_profiles(sub_ids, num_bins, offset, numPart_stars, sub_pos, starType, star_pos, boxSize,\
                                    group_R_Crit200, num_gmcrit200)
    
    return sub_profiles


run = int(sys.argv[1])
basePath = f'/virgotng/universe/IllustrisTNG/TNG50-{run}/output'

#--- parameters ---#
snap1 = 99
snap2 = 33
start_snap = snap1

numBins = 21
num_gmcrit200 = 1
dist_bins = np.linspace(0,num_gmcrit200,numBins)
#------------------#

# load halo data to define halo mass bins
h_const = il.groupcat.loadHeader(basePath, start_snap)['HubbleParam']
groups = il.groupcat.loadHalos(basePath, start_snap, fields = ['Group_M_Crit200','GroupFirstSub','Group_R_Crit200'])
group_masses = groups['Group_M_Crit200']*1e10/h_const

#differentiate between halos of dwarf / milky way / group size
dwarf_ids = np.where(np.logical_and(group_masses > 10**(10.8), group_masses < 10**(11.2)))
mw_ids = np.where(np.logical_and(group_masses > 10**(11.8), group_masses < 10**(12.2)))
group_ids = np.where(np.logical_and(group_masses > 10**(12.6), group_masses < 10**(13.4)))

#find ids of associated centrals
sub_ids_dwarves = groups['GroupFirstSub'][dwarf_ids]
sub_ids_mw = groups['GroupFirstSub'][mw_ids]
sub_ids_groups = groups['GroupFirstSub'][group_ids]

groupRcrit200_dwarves = groups['Group_R_Crit200'][dwarf_ids]
groupRcrit200_mw = groups['Group_R_Crit200'][mw_ids]
groupRcrit200_groups = groups['Group_R_Crit200'][group_ids]

# plot profiles for snapshot 99 (z=0):

star_pos = il.snapshot.loadSubset(basePath, snap1, 4, fields = ['Coordinates'])
subs = il.groupcat.loadSubhalos(basePath, snap1, fields = ['SubhaloPos','SubhaloLenType','SubhaloHalfmassRadType'])
sub_profiles_dwarves = exsitu_radial_profile(basePath, snap1, sub_ids_dwarves, groupRcrit200_dwarves,\
                                             numBins, num_gmcrit200, star_pos, subs)
print('1/6', flush = True)
sub_profiles_mw = exsitu_radial_profile(basePath, snap1, sub_ids_mw, groupRcrit200_mw, numBins, num_gmcrit200,\
                                        star_pos, subs)
print('2/6', flush = True)
sub_profiles_groups = exsitu_radial_profile(basePath, snap1, sub_ids_groups, groupRcrit200_groups,\
                                            numBins, num_gmcrit200, star_pos, subs)
print('3/6', flush = True)

fig, ax = plt.subplots(1,1, figsize = (8,6))
ax.plot(dist_bins, np.nanmedian(sub_profiles_dwarves,axis = 0),label = 'dwarfs', color = 'C0')
ax.plot(dist_bins, np.nanmedian(sub_profiles_mw,axis = 0),label = 'MW-like', color = 'C1')
ax.plot(dist_bins, np.nanmedian(sub_profiles_groups,axis = 0),label = 'groups', color = 'C2')

# ax.fill_between(dist_bins, np.nanpercentile(sub_profiles_dwarves,16,axis = 0), np.nanpercentile(sub_profiles_dwarves,84,axis = 0), alpha = 0.3, color = 'C0')
# ax.fill_between(dist_bins, np.nanpercentile(sub_profiles_mw,16,axis = 0), np.nanpercentile(sub_profiles_mw,84,axis = 0), alpha = 0.3, color = 'C1')
# ax.fill_between(dist_bins, np.nanpercentile(sub_profiles_groups,16,axis = 0), np.nanpercentile(sub_profiles_groups,84,axis = 0), alpha = 0.3, color = 'C2')

dwarfs = mpatches.Patch(color='tab:blue', label='dwarfs')
MWs = mpatches.Patch(color='tab:orange', label='MW-like')
groups = mpatches.Patch(color='tab:green', label='groups')

ax.legend(handles = [dwarfs, MWs, groups], loc = 'upper right')

# plot profiles for snapshot 33 (z=2):

groups = il.groupcat.loadHalos(basePath, snap2, fields = ['Group_M_Crit200','GroupFirstSub','Group_R_Crit200'])
group_masses = groups['Group_M_Crit200']*1e10/h_const

#differentiate between halos of dwarf / milky way / group size
dwarf_ids = np.where(np.logical_and(group_masses > 10**(10.8), group_masses < 10**(11.2)))
mw_ids = np.where(np.logical_and(group_masses > 10**(11.8), group_masses < 10**(12.2)))
group_ids = np.where(np.logical_and(group_masses > 10**(12.6), group_masses < 10**(13.4)))

#find ids of associated centrals
sub_ids_dwarves = groups['GroupFirstSub'][dwarf_ids]
sub_ids_mw = groups['GroupFirstSub'][mw_ids]
sub_ids_groups = groups['GroupFirstSub'][group_ids]

groupRcrit200_dwarves = groups['Group_R_Crit200'][dwarf_ids]
groupRcrit200_mw = groups['Group_R_Crit200'][mw_ids]
groupRcrit200_groups = groups['Group_R_Crit200'][group_ids]

subs = il.groupcat.loadSubhalos(basePath, snap2, fields = ['SubhaloPos','SubhaloLenType','SubhaloHalfmassRadType'])
del star_pos
star_pos = il.snapshot.loadSubset(basePath, 33, 4, fields = ['Coordinates'])
sub_profiles_dwarves = exsitu_radial_profile(basePath, snap2, sub_ids_dwarves, groupRcrit200_dwarves,\
                                             numBins, num_gmcrit200, star_pos, subs)
print('4/6', flush = True)
sub_profiles_mw = exsitu_radial_profile(basePath, snap2, sub_ids_mw, groupRcrit200_mw, numBins, num_gmcrit200,\
                                        star_pos, subs)
print('5/6', flush = True)
sub_profiles_groups = exsitu_radial_profile(basePath, snap2, sub_ids_groups, groupRcrit200_groups,\
                                            numBins, num_gmcrit200, star_pos, subs)
print('6/6', flush = True)

ax.plot(dist_bins, np.nanmedian(sub_profiles_dwarves,axis = 0), color = 'C0', linestyle = 'dotted')
ax.plot(dist_bins, np.nanmedian(sub_profiles_mw,axis = 0), color = 'C1', linestyle = 'dotted')
ax.plot(dist_bins, np.nanmedian(sub_profiles_groups,axis = 0), color = 'C2', linestyle = 'dotted')

#ax.set_yscale('log')
ax.set_xlabel(r'radial distance [$\rm{R}_{\rm 200c}$]')
ax.set_ylabel(r'$M_{\ast, \,\rm insitu}\,([r,\Delta r])\,/\,M_{\ast}\,([r,\Delta r])$')
ax.set_ylim(-0.03,1.03)
fig.tight_layout()

# specify path to your output directory
dirname = 'pics/ex-situ_in-situ_mass_fraction'
assert isdir(dirname), 'Directory does not exist.'
plt.savefig(dirname + 'insitu_frac_rad_profile_' + basePath[35:39] +'.pdf', format = 'pdf')