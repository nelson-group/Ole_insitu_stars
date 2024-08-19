
import sys
sys.path.append('/vera/u/olwitt/illustris_python/illustris_python')
from loadMPBs import loadMPBs
import illustris_python as il
import numpy as np
import h5py
import numba as nb
from numba import jit, njit
from numpy.linalg import inv
import sys
from os.path import isfile, isdir
from os import mkdir
import illustrisFuncs as iF

def save_subhalo_pos(basePath):
    """Save subhalo position trees as a dict with keys 'subhaloID' and values 'SubhaloPos', where
    the latter contains all positions from snap 0 until snap 99 (if available)"""
    sub_flag = il.groupcat.loadSubhalos(basePath,99,fields=['SubhaloFlag'])
    numSubs = sub_flag.shape[0]
    ids = np.arange(numSubs)
    trees = loadMPBs(basePath, 99, ids = ids, fields = ['SubhaloPos'])
    
    #find missing trees
    missing = []
    tree_check = list(trees)
    counter = 0
    for i in range(numSubs):
        if i != tree_check[counter]:
            missing.append(i)
            i+=1
            continue
        counter+=1
    
    #make all trees the same length
    SubhaloPos_allSubs = np.full((numSubs, 100, 3), np.nan)
    
    for i in range(numSubs):        
        if i in missing: #if subhalo hast no tree, skip it and create dataset filled with NaN
            sub_flag[i] = 0
            continue    
        length = trees[i]['count']
        SubhaloPos_allSubs[i,:length,:] = trees[i]['SubhaloPos'][:,:]
    
    #save trees
    # specify path to your directory
    f = h5py.File('files/' + basePath[32:39] + '/SubhaloPos_new.hdf5','w')
    f.create_dataset('SubhaloPos', data = SubhaloPos_allSubs)
    f.create_dataset('SubhaloFlag', data = sub_flag)
    f.close()
    return

@jit(nopython = True)
def linear(x,a,b):
    return a*x+b

@jit(nopython = True)
def linear_fit(x,y):
    A = np.column_stack((np.ones(x.shape[0]), x))
    opt_param = np.ascontiguousarray(inv(A.T@A))@A.T@y
    return opt_param

@jit(nopython = True)
def extrapolate_new(arr, a, boxSize, bound):
    is_extrapolated = np.full(arr.shape[0], False)
    for i in nb.prange(arr.shape[0]): #extrapolate for every subhalo...
        for j in range(3): #... and every coordinate
         
            #if first or second coordinate could not be extrapolated:
            if j > 0 and is_extrapolated[i] == False:
                continue #skip the other ones
            coord = arr[i,:,j]
            
            good_indices = np.arange(coord.size)
            gaps = np.where(np.isnan(coord))[0]
            
            if gaps.size > 80:
                is_extrapolated[i] = False
                continue
            
            top = np.where(coord > (1 - bound) * boxSize)[0]
            bottom = np.where(coord < bound * boxSize)[0]
            
            if top.size > 0 and bottom.size > 0:
                if top.size >= bottom.size:
                    gaps = np.concatenate((gaps, bottom))
                else:
                    gaps = np.concatenate((gaps, top))
            
            good_indices = np.delete(good_indices, gaps)
            
            gap_coord = np.interp(a[gaps],a[good_indices],coord[good_indices])
            coord[gaps] = gap_coord
            new_coord = coord
            
            k = max(good_indices)
            if k < 100:
                #only extrapolate with four or more (good nonzero-) entries
                if(good_indices.size > 5): 
                    popt = linear_fit(a[good_indices[-5:]],coord[good_indices[-5:]])
                    new_coord[k:] = linear(a[k:],popt[1], popt[0])
                        
            
            new_coord[np.where(new_coord > boxSize)] -= boxSize
            new_coord[np.where(new_coord < 0)] += boxSize
            is_extrapolated[i] = True

            arr[i,:,j] = new_coord
            del coord
    return arr, is_extrapolated

#extrapolate positions
def extrapolatePos(basePath, boxSize, a, bound):
    assert isfile('files/'+ basePath[32:39] +'/SubhaloPos_new.hdf5'), 'File does not exist!'
    f = h5py.File('files/'+ basePath[32:39] +'/SubhaloPos_new.hdf5', "r")
    arr = f['SubhaloPos'][:,:,:]
    
    #extrapolate positions and save
    res, is_extrapolated = extrapolate_new(arr,a,boxSize, bound) 

    #specify path to your directory
    result = h5py.File('files/'+ basePath[32:39] +'/SubhaloPos_new_extrapolated.hdf5', 'w')
    result.create_dataset('SubhaloPos',data = res)
    result.create_dataset('is_extrapolated', data = is_extrapolated)
    f.close()
    result.close()
    return

run = int(sys.argv[1])
basePath = f'/virgotng/universe/IllustrisTNG/TNG50-{run}/output'

# save subhalo position histories in a subhalo by subhalo fashion (not snap by snap)
save_subhalo_pos(basePath)
print('Subhalo position histories saved')

boxSize = il.groupcat.loadHeader(basePath,99)['BoxSize']
z = iF.give_z_array(basePath)
a = 1/(1+z)
# percentage of the box size that is considered as boundary
# threshold for extrapolation
bound = 0.1

# extrapolate (strange or missing) positions
extrapolatePos(basePath, boxSize, a, bound)
print('Extrapolation done')