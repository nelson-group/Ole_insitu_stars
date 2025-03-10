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

def give_z_array(basePath):
    z = np.zeros(100)
    for i in range(99,-1,-1):
        z[99-i] = il.groupcat.loadHeader(basePath,i)['Redshift']
    return z

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
    
def halo_form_snap(basePath, snap, central_ids = None, field = 'Group_M_Crit200'):
    start = time.time()
    z = give_z_array(basePath)
    if central_ids is None:
        groupFirstSubs = il.groupcat.loadHalos(basePath, snap, fields = ['GroupFirstSub'])
        central_ids = groupFirstSubs[np.where(groupFirstSubs != -1)[0]]
    trees = loadMPBs(basePath, snap, central_ids, fields = [field])
    end_loading = time.time()
    print('loading done in ',end_loading - start)
    masses = np.empty((central_ids.shape[0],100), dtype = np.float32)
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
            if star_indices.size < 1: #galaxies with zero insitu stars don't have an age
                continue
    
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

@jit(nopython = True)#, parallel = True)
def snap_to_z(z_arr, snap):
    res = np.full(snap.shape[0], np.nan, dtype = np.float32)

    # redshift 0 at index 99
    if np.argmin(z_arr) == 0:
        z_arr = z_arr[::-1]
    
    for i in range(snap.shape[0]):
        # interpolate between the two closest snapshots
        lower = int(snap[i])
        upper = lower + 1

        res[i] = z_arr[lower] + (z_arr[upper] - z_arr[lower]) * (snap[i] - lower)
    return res

@jit(nopython = True)
def find_tracers_of_subs(subs, offsets):
    """Given an array of subhalo ids, computes an array of indices based on the values of offsets."""
    res = np.zeros(offsets[-1], dtype = np.int64)

    counter = 0
    for i in range(subs.shape[0]):
        inds = np.arange(offsets[subs[i]], offsets[subs[i] + 1])
        res[counter:counter + inds.shape[0]] = inds
        counter += inds.shape[0]

    res = res[:counter]
    return res

@jit(nopython = True)
def get_time_difference_in_Gyr(a0, a1):
    omega_lambda = 0.6911
    omega_m = 0.3089
    h = 0.6774
    H0 = 3.2407789e-18 #in s^-1
    
    SEC_PER_MEGAYEAR = 1000000 * 365 * 86400 
    
    factor1 = 2.0 / (3.0 * np.sqrt(omega_lambda))

    term1   = np.sqrt(omega_lambda / omega_m) * a0**1.5
    term2   = np.sqrt(1 + omega_lambda / omega_m * a0**3)
    factor2 = np.log(term1 + term2)

    t0 = factor1 * factor2

    term1   = np.sqrt(omega_lambda / omega_m) * a1**1.5
    term2   = np.sqrt(1 + omega_lambda / omega_m * a1**3)
    factor2 = np.log(term1 + term2)

    t1 = factor1 * factor2

    result = t1 - t0

    time_diff = result / (H0 * h) #now in seconds
    time_diff /= SEC_PER_MEGAYEAR * 1000 #now in gigayears

    return time_diff