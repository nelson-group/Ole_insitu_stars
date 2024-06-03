import time
import illustris_python as il
import numpy as np
import h5py
import numba as nb
from numba import jit, njit
import tracerFuncs as tF
import locatingFuncs as lF
import funcs
from os.path import isfile
import sys
sys.path.append('/vera/u/olwitt/illustris_python/illustris_python')
from loadMPBs import loadMPBs


def group_masses_for_sub_ids(groupMasses, subMasses, groupFirstSub, subIDs):
    all_group_masses = np.zeros(subIDs.shape[0])
    
    #counter starts at first central before the first subID
    halo_counter = int(groupFirstSub[np.max(np.where(groupFirstSub <= subIDs[0])[0])])
    for i in range(subIDs.shape[0]):
        if subIDs[i] == groupFirstSub[halo_counter]:
            all_group_masses[i] = groupMasses[halo_counter]
            halo_counter += 1
            continue
        all_group_masses[i] = subMasses[subIDs[i]]
    
    return all_group_masses

def give_z_array(basePath):
    z = np.zeros(100)
    for i in range(99,-1,-1):
        z[99-i] = il.groupcat.loadHeader(basePath,i)['Redshift']
    return z

#useless function, just use loadTree with the quantity!!
# def give_subhalo_quantity_array(basePath, quantity, subID=-1):
#     assert subID >= 0, 'specify subID!'
#     tree = il.sublink.loadTree(basePath, 99, subID, fields=['SubfinID'], onlyMPB = True)
#     res0 = il.groupcat.loadSingle(basePath, 99, subhaloID = subID)[str(quantity)]
    
#     if res0.size > 1:
#         res = np.empty(shape = (98,res0.size), dtype = res0.dtype)
#     else:
#         res = np.empty(98, dtype = res0.dtype)
#     res[97] = res0
#     for i in range(98,1,-1):
#         if 99 - i < tree['count']: #if tree has sufficient entries
#             target_subID = tree['SubfindID'][start_snap - i]
#             res[i-2] = il.groupcat.loadSingle(basePath, i, subhaloID = target_subID)[str(quantity)]
#         else:
#             res[i-1] = None
    
    return np.flip(res,axis=0)

@jit(nopython = True, parallel = True)
def compute_z_form(central_ids, trees, z, snap):
    numGroups = central_ids.shape[0]
    done = np.ones(numGroups)
    z_form = np.zeros(numGroups)
    for i in nb.prange(numGroups):
        if np.isnan(trees[i]).all():
            done[i] = 0
            z_form[i] = np.nan
            continue
        masses = trees[i,:]
        diff = 1/2 - masses/masses[0]
        ind = np.where(diff[1:] * diff[:-1] < 0)[0]
        ind2 = ind[0]
        z_form[i] = z[int(ind2)]
    return done, z_form 
    
def halo_form_snap(basePath, snap, central_ids = 1, field = 'Group_M_Crit200'):
    start = time.time()
    z = give_z_array(basePath)
    if type(central_ids) == int:
        groupFirstSubs = il.groupcat.loadHalos(basePath, snap, fields = ['GroupFirstSub'])
        central_ids = groupFirstSubs[np.where(groupFirstSubs != -1)[0][:central_ids]]
    trees = loadMPBs(basePath, snap, central_ids, fields = [field])
    end_loading = time.time()
    print('loading done in ',end_loading - start)
    masses = np.empty((central_ids.shape[0],100))
    check = list(trees)
    missing = []
    counter = 0
    for i in central_ids:
        if i != check[counter]:
            missing.append(i)
            i+=1
            continue
        counter+=1
    print('# missing trees: ',len(missing))
    print('# remaining trees: ',central_ids.shape[0] - len(missing))
    for i in range(central_ids.shape[0]):
        masses[i,:] = np.full(100, np.nan)
        if central_ids[i] in missing:
            continue
        count = trees[central_ids[i]]['count']
        if field == 'SubhaloMassInRadType':
            masses[i,:count] = trees[central_ids[i]][field][:,1]
        else:
            masses[i,:count] = trees[central_ids[i]][field][:]
    end_conv = time.time()
    print('converting done in ',end_conv - end_loading)
    done, z_form = compute_z_form(central_ids, masses, z, snap)
    end_comp = time.time()
    print('computing done in ',end_comp - end_conv)
    end = time.time()
    print('done in ', end - start)
    return done.astype(int), z_form

@jit(nopython = True, parallel = True)
def compute_ages(subIDs, starsInSubOffset, numStarsInSubs, insitu, onlyInsitu, star_formation_time):
    galaxy_form_z = np.full(subIDs.shape[0],np.nan)
    for i in nb.prange(0,subIDs.shape[0]):
        star_indices = np.arange(starsInSubOffset[subIDs[i]],starsInSubOffset[subIDs[i]] +\
                                 numStarsInSubs[subIDs[i]])
        if star_indices.size < 1: #galaxies with zero stars don't have an age
            continue
        
        if onlyInsitu:
            star_indices = star_indices[np.nonzero(insitu[star_indices])]
    
        galaxy_form_a = np.nanmedian(star_formation_time[star_indices])
        galaxy_form_z[i] = 1/galaxy_form_a - 1
    
    return galaxy_form_z

def galaxy_ages(basePath, snap, onlyCentrals = False, onlyInsitu = False):
    start = time.time()
    
    g = h5py.File(basePath[:-6] + 'postprocessing/offsets/offsets_' + str(snap).zfill(3) + '.hdf5','r')
    starsInSubOffset = g['Subhalo/SnapByType'][:,4]
    g.close()
    numStarsInSubs = il.groupcat.loadSubhalos(basePath, snap, fields = ['SubhaloLenType'])[:,4]
    
    if onlyCentrals:
        subIDs = il.groupcat.loadHalos(basePath, snap, fields = ['GroupFirstSub'])
        subIDs = np.unique(subIDs[np.where(subIDs != -1)])
    else:
        subIDs = np.arange(numStarsInSubs.shape[0])
    
    check = h5py.File(basePath[:-6] + 'postprocessing/StellarAssembly/stars_' + str(snap).zfill(3) + '.hdf5','r')
    insitu = check['InSitu'][:] #1 if star is formed insitu, 0 if it was formed ex-situ and -1 otherwise (fuzz)
    insitu = np.asarray(insitu == 1)
    check.close()
    
    star_formation_time = il.snapshot.loadSubset(basePath, snap, 4, fields=['GFM_StellarFormationTime'])
    star_formation_time[np.where(star_formation_time < 0)] = np.nan
    
    galaxy_form_z = compute_ages(subIDs, starsInSubOffset, numStarsInSubs, insitu, onlyInsitu, star_formation_time)
        
    end = time.time()
    print('done in ', end - start)
    return galaxy_form_z

def plot_vs_galaxy_mass(values, run, snap, f, f_args = {}, mask = None):
    '''
    Bins data given for every subhalo w.r.t. galaxy stellar mass.
    '''
    basePath = f'/virgotng/universe/IllustrisTNG/TNG50-{run}/output'
    header = il.groupcat.loadHeader(basePath, snap)
    h_const = header['HubbleParam']
    if mask is None:
        num_subs = header['Nsubgroups_Total']
        mask = np.arange(num_subs)
    
    stellar_masses = il.groupcat.loadSubhalos(basePath, snap, fields = ['SubhaloMassInRadType'])[mask,4] * 1e10/h_const
    
    values = values[mask]
    
    nonzero = np.nonzero(stellar_masses)[0]
    masses = np.log10(stellar_masses[nonzero])
    values = values[nonzero]
    
    assert values.shape[0] == masses.shape[0], 'something went wrong'
    
    res = f(masses, values ,**f_args)
    
    mass_bins = res[0]
    value_bins = res[1]

    low = res[2]
    up = res[3]
    
    return mass_bins, value_bins, low, up
