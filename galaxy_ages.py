import illustris_python as il
import numpy as np
import h5py
import illustrisFuncs as iF
from os.path import isfile
import sys

run = int(sys.argv[1])
snap = int(sys.argv[2])
# specify, if you want to only consider central galaxies (1 = True, 0 = False)
onlyCentrals = int(sys.argv[3]) == 1

# specify, if you want to only consider in-situ stars (1 = True, 0 = False)
onlyInsitu = int(sys.argv[4]) == 1

print(onlyCentrals, onlyInsitu)

basePath='/virgotng/universe/IllustrisTNG/TNG50-' + str(run) + '/output'
h_const = il.groupcat.loadHeader(basePath,snap)['HubbleParam']
boxSize = il.groupcat.loadHeader(basePath,snap)['BoxSize']

galaxy_form_z = iF.galaxy_ages(basePath,snap, onlyCentrals, onlyInsitu)

filename = 'files/' + basePath[32:39] + f'/galaxy_ages_snap{snap}.hdf5'
if onlyCentrals == True:
    add = '_centrals'
    if onlyInsitu == True:
        add = add + '_insitu'
        onlyInsitu = False
    filename = filename[:25] + add + filename[25:]

if onlyInsitu == True:    
    filename = filename[:25] + '_insitu' + filename[25:]  

f = h5py.File(filename,'w')
f.create_dataset(f'galaxy_stellar_formation_redshift_snap{snap}',data = galaxy_form_z)
f.close()