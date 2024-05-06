import dm
import time
import illustris_python as il
import numpy as np
import h5py
import numba as nb
from numba import jit, njit
import tracerFuncs as tF
import funcs
from os.path import isfile, isdir
import sys

sys.path.append('/vera/u/olwitt/illustris_python/illustris_python')
from loadMPBs import loadMPBs

#---- settings----#
run = int(sys.argv[1])
stype = 'dm'
halo_type = str(sys.argv[2])
target_snap = int(sys.argv[3])
cut_snap = int(sys.argv[4])
basePath='/virgotng/universe/IllustrisTNG/TNG50-' + str(run) + '/output'
start_snap = 99
start = time.time()

assert isdir('/vera/ptmp/gc/olwitt/' + stype + '/'+basePath[32:39]+'/lagrangian_regions')

h_const = il.groupcat.loadHeader(basePath,99)['HubbleParam']
boxSize = il.groupcat.loadHeader(basePath,99)['BoxSize']

#introduce mass bins:
groups = il.groupcat.loadHalos(basePath, start_snap, fields = ['Group_M_Crit200','GroupFirstSub'])
group_masses = groups['Group_M_Crit200']*1e10/h_const

#differentiate between halos of dwarf / milky way / group size
dwarf_ids = np.where(np.logical_and(group_masses > 10**(10.8), group_masses < 10**(11.2)))[0]
mw_ids = np.where(np.logical_and(group_masses > 10**(11.8), group_masses < 10**(12.2)))[0]
group_ids = np.where(np.logical_and(group_masses > 10**(12.6), group_masses < 10**(13.4)))[0]
all_ids = np.arange(group_masses.shape[0])

#find ids of associated centrals
sub_ids_dwarves = groups['GroupFirstSub'][dwarf_ids]
sub_ids_mw = groups['GroupFirstSub'][mw_ids]
sub_ids_groups = groups['GroupFirstSub'][group_ids]
all_central_ids = groups['GroupFirstSub'][:]

# sub_positions = h5py.File('files/'+basePath[32:39]+'/SubhaloPos_new_extrapolated.hdf5','r') 
# is_extrapolated = sub_positions['is_extrapolated'][:]
# sub_positions.close()

sub_ids = sub_ids_mw if halo_type == 'mw' else sub_ids_dwarves if halo_type == 'dwarves' else sub_ids_groups if halo_type == 'groups' else all_central_ids

ids = mw_ids if halo_type == 'mw' else dwarf_ids if halo_type == 'dwarves' else group_ids if halo_type == 'groups' else all_ids

#Filter out halos without any subhalo
sub_ids = sub_ids[np.where(sub_ids != -1)[0]]

r_norm_trees = loadMPBs(basePath, start_snap, ids = sub_ids, fields = ['SubhaloHalfmassRadType','SubfindID'])

end = time.time()
print('time for initial loading: ',end-start)
del end,start

sub_medians, sub_medians_r_vir, extrapolated_sub_ids, extrapolated_central_sub_ids = \
dm.lagrangian_region(basePath, stype, start_snap, target_snap, cut_snap, sub_ids, boxSize, r_norm_trees)

f = h5py.File('/vera/ptmp/gc/olwitt/' + stype + '/'+basePath[32:39]+f'/lagrangian_regions/lagrangian_regions_cut{cut_snap}_{target_snap}.hdf5','w')
f.create_dataset('lagrangian_regions_shmr',data = sub_medians)
f.create_dataset('lagrangian_regions_r_vir',data = sub_medians_r_vir)

f.create_dataset('which_galaxy_ids', data = extrapolated_sub_ids)
f.create_dataset('which_central_galaxy_ids', data = extrapolated_central_sub_ids)
f.close()

