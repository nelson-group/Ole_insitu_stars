import time
import illustris_python as il
import numpy as np
import h5py
import numba as nb
from numba import jit, njit
import tracerFuncs as tF
import locatingFuncs as lF
import illustrisFuncs as iF
import funcs
from os.path import isfile

import sys
sys.path.append('/vera/u/olwitt/illustris_python/illustris_python')
from loadMPBs import loadMPBs

run = int(sys.argv[1])

basePath='/virgotng/universe/IllustrisTNG/TNG50-' + str(run) + '/output'
h_const = il.groupcat.loadHeader(basePath,99)['HubbleParam']
boxSize = il.groupcat.loadHeader(basePath,99)['BoxSize']

#introduce mass bins:
groups = il.groupcat.loadHalos(basePath, 99, fields = ['GroupFirstSub'])

#find ids of associated centrals

all_central_ids = groups[:]

z = iF.give_z_array(basePath)
field = 'SubhaloMassInRadType'
done, z_form = iF.halo_form_snap(basePath,99,all_central_ids[np.where(all_central_ids != -1)[0]], field = field)

f = h5py.File('files/' + basePath[32:39] + f'/halo_formation_times_{field}.hdf5','w')
ds = f.create_dataset('Done', data = done)
ds2 = f.create_dataset('formation_redshift',data = z_form)
f.close()