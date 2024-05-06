import numpy as np
from numba import jit, njit
import numba as nb
import illustris_python as il
import h5py

def is_insitu(basePath,star_indices,snap=99):
    #load postprocessing file that contains information about insitu status of stars in snapshot
    check = h5py.File(basePath[:-6] + 'postprocessing/StellarAssembly/stars_0' + str(snap) + '.hdf5','r')
    #determine for every star, if it was created insitu
    insitu = np.asarray(check['InSitu'][star_indices]==1)
    check.close()
    return insitu

@njit
def dist(x,y,boxSize):
    diff = x-y
    diff[np.where(diff>boxSize/2)] -= boxSize
    diff[np.where(diff<=-boxSize/2)] += boxSize
    r=np.linalg.norm(diff)
    return r

def binData(val, numBins):
    
    minVal = min(val)
    maxVal = max(val)
    
    binWidth = (maxVal - minVal) / numBins
    
    frac = np.zeros(numBins)
    bins = np.linspace(minVal,maxVal,numBins)
    
    for b in range(numBins):
        indices=np.where(np.logical_and(val >= minVal + b*binWidth, val < minVal + (b+1)*binWidth))[0]
        frac[b]=indices.size      
    return bins,frac

def binData_w_bins(val, bins): 
    num = np.zeros(bins.size)
    for b in range(bins.size):
        if b < bins.size-1:
            indices = np.where(np.logical_and(val >= bins[b], val < bins[b+1]))[0]
        else:
            indices = np.where(val > bins[-1])[0]
        num[b] = indices.size      
    return num