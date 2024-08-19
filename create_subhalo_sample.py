import illustris_python as il
import numpy as np
import h5py
import numba as nb
from numba import jit, njit
import tracerFuncs as tF
from os.path import isfile, isdir
from os import mkdir
import sys

sys.path.append('/vera/u/olwitt/illustris_python/illustris_python')
from loadMPBs import loadMPBs

@jit(nopython = True, parallel = True)
def isSubGalaxy(sub_ids, final_offsets, tracer_cut):
    """ Checks the number of tracers in each galaxy and marks the ones with less than a certain amount of tracers."""
    
    #assume all subhalos have 0 tracers...
    noGalaxy = np.ones(sub_ids.shape[0], dtype = np.ubyte)
    for i in nb.prange(sub_ids.shape[0]):
        sub_id = sub_ids[i]
        # ... and mark the ones with more than 0 tracers as not 'noGalaxy'
        if final_offsets[sub_id + 1] - final_offsets[sub_id] < tracer_cut:
            continue
        noGalaxy[i] = 0
    return noGalaxy

def create_subhalo_sample(basePath, stype, start_snap, tracer_cut = 1):
    """Creates a sample of subhalos based on their amount of in-situ tracers at z=0, their position history
    and their mass at z=0."""
    header = il.groupcat.loadHeader(basePath,start_snap)
    num_subs = header['Nsubgroups_Total']
    h_const = header['HubbleParam']

    #introduce mass bins (just for analysis, not for computation):
    central_ids = il.groupcat.loadHalos(basePath, start_snap, fields = ['GroupFirstSub'])

    sub_ids = np.arange(num_subs)

    #Filter out halos without any subhalo and satellites
    subhaloFlag = np.zeros(num_subs, dtype = np.ubyte)
    subhaloFlag[central_ids[np.where(central_ids != -1)]] = 1
    
    #load MPB trees
    trees = loadMPBs(basePath, start_snap, ids = sub_ids, fields = ['SubfindID'])
    
    #load data from files ---------------------------------------------------------------------------------
    assert isfile('files/'+basePath[32:39]+'/SubhaloPos_new_extrapolated.hdf5')
    sub_positions = h5py.File('files/'+basePath[32:39]+'/SubhaloPos_new_extrapolated.hdf5','r') 
    is_extrapolated = sub_positions['is_extrapolated'][:]
    sub_positions.close()
    
    if stype.lower() in ['insitu', 'exsitu']:
        file = '/vera/ptmp/gc/olwitt/' + stype + '/' + basePath[32:39] + f'/parent_indices_{start_snap}.hdf5'
        assert isfile(file)
        f = h5py.File(file,'r')
    
    #offsets -----------------------------------------------------------------------------------------------
        #here, it's okay that the offsets at the start snapshot are used as they are identical at every snapshot
        if f.__contains__(f'snap_{start_snap}/numTracersInParents'):
            numTracersInParents = f[f'snap_{start_snap}/numTracersInParents'][:]
        else:
            numTracersInParents = f[f'snap_{start_snap}/tracers_in_parents_offset'][:]
        f.close()
        
        if stype == 'insitu':
            insituStarsInSubOffset = tF.insituStarsInSubOffset(basePath,start_snap)
        else:
            insituStarsInSubOffset = tF.exsituStarsInSubOffset(basePath,start_snap)
            
        final_offsets = tF.tracersInSubhalo(insituStarsInSubOffset,numTracersInParents).astype(int)
        final_offsets = np.insert(final_offsets,0,0)
        
        del insituStarsInSubOffset, numTracersInParents
        
    elif stype.lower() in ['dm', 'darkmatter', 'dark_matter', 'dark matter']:
        stype = 'dm'
        
        file = '/vera/ptmp/gc/olwitt/' + stype + '/' + basePath[32:39] + f'/dm_indices_{start_snap}.hdf5'
        f = h5py.File(file,'r')        
        final_offsets = f['dmInSubOffset'][:]
        f.close()
        
    else:
        raise Exception('Invalid star/particle type!')
    # ^ offsets for the parent index table, that's why at snapshot 99
    
    
    
    #which galaxies? ----------------------------------------------------------------------------------------
    not_extrapolated = np.nonzero(np.logical_not(is_extrapolated))[0]
    subhaloFlag[not_extrapolated] = 0
    
    del is_extrapolated, not_extrapolated
    #test, which galaxies have zero tracers of insitu stars
    # (this already excludes all galaxies without any stars, since they can't have insitu stars)    
    noGalaxy = isSubGalaxy(sub_ids, final_offsets, tracer_cut)
    
    #only use galaxies that have at least one tracer particle (at z=0) AND have an extrapolated SubhaloPos entry
    subhaloFlag[np.nonzero(noGalaxy)[0]] = 0
    print(f'# galaxies with at least {tracer_cut} tracer particle(s) AND extrapolated position history: ',\
           np.count_nonzero(subhaloFlag))
    del noGalaxy
    
    #tree check: find missing trees and delete according galaxies from the flag
    missing = []
    counter = 0
    tree_check = list(trees)
    for i in range(num_subs):
        if i != tree_check[counter]:
            missing.append(i)
            i += 1
            continue
        counter += 1
        
    for i in range(sub_ids.shape[0]):
        if sub_ids[i] in missing or sub_ids[i] >= num_subs:
            subhaloFlag[i] = 0
            
    print('# galaxies with trees: ', np.count_nonzero(subhaloFlag))

    # introduce mass cut: consider only galaxies with stellar mass > 10^8.5 Msun at z=0
    stellar_masses = il.groupcat.loadSubhalos(basePath, start_snap, fields = ['SubhaloMassInRadType'])[:,4]
    mass_mask = np.where(stellar_masses * 1e10 / h_const < 10**(8.5))[0]
    subhaloFlag[mass_mask] = 0

    print('# galaxies above logM_star = 8.5: ', np.where(subhaloFlag == 1)[0].shape[0])
        
    return subhaloFlag

#---- settings----#
run = int(sys.argv[1])
stype = str(sys.argv[2])
basePath = f'/virgotng/universe/IllustrisTNG/TNG50-{run}/output'
start_snap = 99

# specify the amount of tracers a galaxy needs to be considered
tracer_cut = 1000

subhaloFlag = create_subhalo_sample(basePath, stype, start_snap, tracer_cut)

dir_name = '/vera/ptmp/gc/olwitt/auxCats/' + basePath[32:39]
if not isdir(dir_name):
    print('Directory does not exist, creating it...')
    mkdir(dir_name)

filename = dir_name + '/subhaloFlag_' + stype + '.hdf5'

f = h5py.File(filename,'w')
f.create_dataset('subhaloFlag', data = subhaloFlag)
f.close()