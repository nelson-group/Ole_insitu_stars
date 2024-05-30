import sys

import illustris_python as il
import numpy as np
import h5py
import numba as nb
from numba import jit, njit
import dm

import funcs

run = int(sys.argv[1])
stype = 'dm'
basePath=f'/virgotng/universe/IllustrisTNG/TNG50-{run}/output'
start_snap = 99
h_const = il.groupcat.loadHeader(basePath, start_snap)['HubbleParam']
#introduce mass bins:
groups = il.groupcat.loadHalos(basePath, start_snap, fields = ['Group_M_Crit200','GroupFirstSub'])
group_masses = groups['Group_M_Crit200']*1e10/h_const

#differentiate between halos of dwarf / milky way / group size
group_ids = np.where(np.logical_and(group_masses > 10**(12.6), group_masses < 10**(13.4)))[0]
mw_ids = np.where(np.logical_and(group_masses > 10**(11.8), group_masses < 10**(12.2)))[0]
dwarf_ids = np.where(np.logical_and(group_masses > 10**(10.8), group_masses < 10**(11.2)))[0]

#find ids of associated centrals
sub_ids_groups = groups['GroupFirstSub'][group_ids]
sub_ids_mws = groups['GroupFirstSub'][mw_ids]
sub_ids_dwarfs = groups['GroupFirstSub'][dwarf_ids]
all_central_ids = groups['GroupFirstSub'][:]

subs = il.groupcat.loadSubhalos(basePath, 99, fields = ['SubhaloFlag'])
numSubs = subs.shape[0]
del subs
sub_ids = np.arange(numSubs)

#giants:
# sub_ids = all_central_ids[np.where(all_central_ids < np.min(sub_ids_groups))]
mp, igm, sub, satellites, mp_satellites, other_centrals, total, nums, z, gal_comp, isGalaxy, directly_from_igm, stripped_from_halos, from_other_halos, mergers, long_range_wind_recycled, nep_wind_recycled, gal_accretion_channels =\
dm.dm_accretion_channels(basePath, stype, sub_ids, start_snap)

f = h5py.File('files/' + basePath[32:39] + '/accretion_channels_' + stype + '.hdf5','w')

f.create_dataset('main_progenitor', data = mp)
f.create_dataset('IGM', data = igm)
f.create_dataset('all_galaxies', data = sub)
f.create_dataset('satellites', data = satellites)
f.create_dataset('satellites_of_main_progenitor', data = mp_satellites)
f.create_dataset('other_centrals', data = other_centrals)

f.create_dataset('totals', data = total)
f.create_dataset('binned_values', data = nums)
f.create_dataset('redshift', data = z)
f.create_dataset('galaxy_composition', data = gal_comp)

grp = f.create_group('subhalo_ids')
grp.create_dataset('all_subhalo_ids', data = sub_ids[np.nonzero(isGalaxy)[0]])
grp.create_dataset('subhalo_ids_dwarfs', data = sub_ids_dwarfs)
grp.create_dataset('subhalo_ids_mws', data = sub_ids_mws)
grp.create_dataset('subhalo_ids_groups', data = sub_ids_groups)

grp2 = f.create_group('accretion_channels')
channels = np.array([directly_from_igm, from_other_halos, mergers, stripped_from_halos, long_range_wind_recycled, nep_wind_recycled])
grp2.create_dataset('overall_fractions',data = channels)
grp2.create_dataset('for_all_galaxies', data = gal_accretion_channels)
f.close()