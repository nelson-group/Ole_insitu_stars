import numba as nb
import numpy as np
from numba import jit, njit
import h5py
import os

from collections import deque
from bisect import insort, bisect_left
from itertools import islice

#@njit
def log10_mod(arr):
    if not isinstance(arr, np.ndarray):
        return np.log10(arr) if arr > 0 else np.nan
    else:
        res = np.empty(arr.shape, dtype = float)
        for ind, val in np.ndenumerate(arr):
            res[ind] = np.log10(val) if val > 0 else np.nan
        return res

def print_array_1d(arr):
    print('[',end='')
    for i in range(arr.shape[0]):
        print(arr[i],end = ', ')
    print(']')
    return

def find_file_with_lowest_number(directory):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and\
             os.path.getsize(os.path.join(directory, f)) > 1024]
    min_number = float('inf')
    min_file = None

    for file in files:
        try:
            number = int(file.split('.')[0].split('_')[-1])
#             print(number, end = ' ')
            if number < min_number:
                min_number = number
                min_file = file
        except ValueError:
            pass

    return min_number

#faster than np.isin, bc. np.isin not supported from numba
@njit(parallel=True)
def isin(a, b):
    out=np.empty(a.shape[0], dtype=nb.boolean)
    b = set(b)
    for i in nb.prange(a.shape[0]):
        if a[i] in b:
            out[i]=True
        else:
            out[i]=False
    return out

@njit
def where_one(a,b):
    for i in nb.prange(b.shape[0]):
        if a == b[i]:
            return i
    return -1

@njit(parallel=True)
def is_unique(a):
    for i in range(a.size):
        index = np.where(a[i]==a[i:],1,0)
        if(np.sum(index)>1):
            return False
    return True

@njit
def is_sorted_increasing(a):
    for i in range(a.size-1):
        if(a[i]>a[i+1]):
            return False
    return True

@njit
def is_sorted_decreasing(a):
    for i in range(a.size-1):
        if(a[i]<a[i+1]):
            return False
    return True

@njit
def insert(arr,index,value):
    assert index<=arr.size
    out = np.zeros(arr.size+1, dtype = arr.dtype)
    h=0
    for i in nb.prange(out.size):
        if i == index:
            out[i]=value
        else:
            out[i]=arr[h]
            h=h+1
    return out

def areEqual(A, B):
    n = len(A)
    if (len(B) != n):
        return False 

    # Create a hash table (dictionary) to count number of instances

    m = {} 
    # For each element of A increase it's instance by 1.
    for i in range(n):
        if A[i] not in m:
            m[A[i]] = 1
        else:
            m[A[i]] += 1         
    # For each element of B decrease it's instance by 1.
    for i in range(n):
        if B[i] in m:
            m[B[i]] -= 1
    # Iterate through map and check if any entry is non-zero
    for i in m:
        if (m[i] != 0):
            return False         
    return True

@njit
def dist(x,y,boxSize):
    diff = x-y
    diff[np.where(diff>boxSize/2)[0]] -= boxSize
    diff[np.where(diff<=-boxSize/2)[0]] += boxSize
    r=np.linalg.norm(diff)
    return r

def dist_vector(x,y,boxSize):
    """Determine the distance of every 3-vector-entry in y to x"""
    assert x.shape == (3,), 'x must be of shape (3,)'
    assert y.shape[1] == 3, 'y must be an array of 3-vectors'

    diff = x - y
    diff[np.where(diff > boxSize/2)[0]] -= boxSize
    diff[np.where(diff <= -boxSize/2)[0]] += boxSize
    r = np.linalg.norm(diff, axis = 1)
    return r

@jit(nopython = True, parallel = True)
def dist_vector_nb(x,y,boxSize):
    """Determine the distance of every 3-vector-entry in y to x"""
#     assert x.shape == (3,), 'x must be of shape (3,)'
#     assert y.shape[1] == 3, 'y must be an array of 3-vectors'
    diff = x - y
    diff[np.where(diff > boxSize/2)[0]] -= boxSize
    diff[np.where(diff <= -boxSize/2)[0]] += boxSize
    r = norm_axis1(diff)
    return r


@jit(nopython=True, parallel = True)
def norm_axis1(arr):
    """Computes the norm of a 2-d array along the second axis."""
    assert len(arr.shape) == 2
    norms = np.empty(arr.shape[0], dtype = arr.dtype)
    for i in nb.prange(arr.shape[0]):
        norms[i] = np.sqrt(np.sum(arr[i,:]*arr[i,:]))
    return norms

def binData(val, numBins, mode = 'num', mini = None, maxi = None):
    """
    input:
    
    val: data to be binned
    numBins: number of bins
    mode: which function to use to combine data in each bin
        options: num: count of data points in bin (default)
                 frac: fraction of total number of data points in bin
                 sum: sum of all values of the data points in bin
                 sum_logs: sum of 10**val of all values in bin
                 median: median value of all data points in bin (ignoring NaN)
                 
    mini: lower boundary of leftmost bin (default: None)
    maxi: upper boundary of rightmost bin (default: None)
    
    output:
    
    bins: the bins as a linspace specified by either the data itself or mini and maxi
    frac: the result array, i.e. the result for each bin
    """
    
    
    if mini is None:
        minVal = np.nanmin(val)
        maxVal = np.nanmax(val)
    else:
        if mini < maxi and mini is not None and maxi is not None:
            minVal = mini
            maxVal = maxi
        else:
            raise ValueError("Specify finite minimum and maximum values with min < max !")
    
    binWidth = (maxVal - minVal) / numBins
    
    frac = np.zeros(numBins)
    bins = np.linspace(minVal,maxVal,numBins)
    
    for b in range(numBins):
        indices=np.where(np.logical_and(val >= minVal + b*binWidth, val < minVal + (b+1)*binWidth))[0]
        if mode == 'num':
            frac[b] = indices.size
        elif mode == 'frac':
            frac[b] = indices.size / val.size
        elif mode == 'sum':
            frac[b] = np.sum(val[indices])
        elif mode == 'sum_logs':
            frac[b] = np.sum(10**val[indices])
        elif mode == 'median':
            frac[b] = np.nanmedian(val[indices])
        elif mode == 'mean':
            frac[b] = np.nanmean(val[indices])
        else:
            frac[b] = np.nan

    return bins,frac

@jit(nopython = True, parallel = True)
def binData_w_bins(val, bins): 
    num = np.full(bins.size, np.nan)
    for b in nb.prange(bins.size):
        if b < bins.size-1:
            indices = np.where(np.logical_and(val >= bins[b], val < bins[b+1]))[0]
        else:
            indices = np.where(val > bins[-1])[0]
            
        if indices.size > 0:
            num[b] = indices.size      
    return num

# def binData_med(xVal, yVal, numBins=4):
#     """
#     Compute a running median from a set of (x,y) points.

#     Input:
#     xVal: Array of x values.
#     yVal: Array of y values.
#     numBins: Number of bins (optional; set to 4 by default).

#     Output:
#     xMed: Array of (median) x values.
#     yMed: Array of (median) y values.
#     """    
#     minVal = np.min(xVal)
#     maxVal = np.max(xVal)
    
#     binWidth = (maxVal - minVal) / numBins
    
#     xMed = np.full(numBins, np.nan)
#     yMed = np.full(numBins, np.nan)
#     y16 = np.full(numBins, np.nan)
#     y84 = np.full(numBins, np.nan)
    
#     for j in range(numBins):
#         relInd = np.where( (xVal >= minVal + j*binWidth) & (xVal < minVal + (j+1)*binWidth) )[0]
#         if(relInd.size>0):
#             xMed[j] = np.nanmedian(xVal[relInd])
#             yMed[j] = np.nanmedian(yVal[relInd])
#             y16[j] = np.nanpercentile(yVal[relInd],16)
#             y84[j] = np.nanpercentile(yVal[relInd],84)
            
#     return xMed, yMed, y16, y84

# def binData_num(xVal, yVal, numBins=4):
# =======
#         num[b] = indices.size      
#     return num

def binData_med(xVal, yVal, numBins=4):
    """
    Compute a running median from a set of (x,y) points.

    Input:
    xVal: Array of x values.
    yVal: Array of y values.
    numBins: Number of bins (optional; set to 4 by default).

    Output:
    xMed: Array of (median) x values.
    yNum: Array of y values.
    """    
    minVal = np.min(xVal)
    maxVal = np.max(xVal)
    
    binWidth = (maxVal - minVal) / numBins
    
    xMed = np.full(numBins, np.nan)
    yNum = np.full(numBins, np.nan)
    
    for j in range(numBins):
        relInd = np.where( (xVal >= minVal + j*binWidth) & (xVal < minVal + (j+1)*binWidth) )[0]
        if(relInd.size>0):
            xMed[j] = np.nanmedian(xVal[relInd])
            yNum[j] = relInd.size
            
    return xMed, yNum

def binData_mean(xVal, yVal, numBins=4):
    """
    Compute a running mean from a set of (x,y) points.

    Input:
    xVal: Array of x values.
    yVal: Array of y values.
    numBins: Number of bins (optional; set to 4 by default).

    Output:
    xMed: Array of (median) x values.
    yNum: Array of (mean) y values.
    """    
    minVal = np.min(xVal)
    maxVal = np.max(xVal)
    
    binWidth = (maxVal - minVal) / numBins
    
    xMed = np.full(numBins, np.nan)
    yMean = np.full(numBins, np.nan)
    
    for j in range(numBins):
        relInd = np.where( (xVal >= minVal + j*binWidth) & (xVal < minVal + (j+1)*binWidth) )[0]
        if(relInd.size>0):
            xMed[j] = np.nanmedian(xVal[relInd])
            yMean[j] = np.nanmean(yVal[relInd])
            
    return xMed, yMean

def binData_med(xVal, yVal, numBins=4, lower = 16, upper = 84):
    """
    Compute a running median from a set of (x,y) points.

    Input:
    xVal: Array of x values.
    yVal: Array of y values.
    numBins: Number of bins (optional; set to 4 by default).
    lower: percentile for lower error line.
    upper: percentile for upper error line.

    Output:
    xMed: Array of (median) x values.
    yMed: Array of (median) y values.
    ylow: Array of y values (lower th percentile).
    yup: Array of y values (upper th percentile).
    yMed: Array of (median) y values.
    """    
    
    minVal = np.min(xVal)
    maxVal = np.max(xVal)
    
    binWidth = (maxVal - minVal) / numBins
    
    xMed = np.full(numBins, np.nan)
    yMed = np.full(numBins, np.nan)
    ylow = np.full(numBins, np.nan)
    yup = np.full(numBins, np.nan)
    
    for j in range(numBins):
        relInd = np.where( (xVal >= minVal + j*binWidth) & (xVal < minVal + (j+1)*binWidth) )[0]
        if(relInd.size>0):
            xMed[j] = np.nanmedian(xVal[relInd])
            yMed[j] = np.nanmedian(yVal[relInd])
            ylow[j] = np.nanpercentile(yVal[relInd],lower)
            yup[j] = np.nanpercentile(yVal[relInd],upper)
            
    return xMed, yMed, ylow, yup

# def is_insitu(basePath, snap):
#     if snap > 9:
#         str_snap = str(snap)
#     else:
#         str_snap = f'0{snap}'
#     check = h5py.File(basePath[:-6] + 'postprocessing/StellarAssembly/stars_0' + str_snap + '.hdf5','r')
#     insitu = check['InSitu'][:] #1 if star is formed insitu and 0 otherwise
#     check.close()
#     return insitu

def running_median_insort(seq, window_size):
    """Contributed by Peter Otten"""
    seq = iter(seq)
    d = deque()
    s = []
    result = []
    for item in islice(seq, window_size):
        d.append(item)
        insort(s, item)
        result.append(s[len(d)//2])
    m = window_size // 2
    for item in seq:
        old = d.popleft()
        d.append(item)
        del s[bisect_left(s, old)]
        insort(s, item)
        result.append(s[m])
    return np.array(result)

@jit(nopython=True, parallel=True)
def take_nb_clip(arr, idxs):
    res = np.empty((idxs.size,), arr.dtype)
    lastIdx = arr.size - 1
    for i in nb.prange(idxs.size):
        idx = idxs[i]
        if idx > lastIdx:
            idx = lastIdx
        elif idx < 0:
            idx = 0
        res[i] = arr[idx]
    return res

@jit(nopython=True, parallel=True)
def searchsorted_nb_left(a, b):
    res = np.empty(len(b), np.intp)
    for i in nb.prange(len(b)):
        res[i] = np.searchsorted(a, b[i], side='left')
    return res

@jit(nopython=True, parallel=True)
def searchsorted_nb_right(a, b):
    res = np.empty(len(b), np.intp)
    for i in nb.prange(len(b)):
        res[i] = np.searchsorted(a, b[i], side='right')
    return res 

@jit(nopython=True)
def insert00(arr):
    res = np.empty(len(arr)+1, dtype = arr.dtype)
    res[0] = 0
    res[1:] = arr
    return res 

@jit(nopython = True, parallel = True)
def data_intersect_value(y,value):
    """Find the intersection of e.g. a profile with a certain value."""
    # test, whether there is an exact point where y is equal to value
    # (in this case the following computation would fail)
    ind = np.where(y == value)[0]
    if ind.size > 0:
        return ind, False
    y_dist = value - y
    #if profile doesn't reach value choose closest value
    if np.all(y < value) or np.all(y > value):
        return np.where(np.min(np.abs(y_dist)) == np.abs(y_dist))[0], False
    #else choose values, where following values are on opposite side of the 'line'
    ind = np.where(y_dist[1:] * y_dist[:-1] < 0)[0]
    return ind, True

@jit(nopython = True, parallel = True)
def hmr(res, cumsum_profiles, dist_bins):
    for i in nb.prange(cumsum_profiles.shape[0]):
        for j in nb.prange(cumsum_profiles.shape[1]): #for all remaining centrals
            if np.all(cumsum_profiles[i,j,:] == -1):
                res[i,j] = -1
                continue
            ind, next_too = data_intersect_value(cumsum_profiles[i,j,:],0.5)
            if not next_too:
                res[i,j] = np.mean(dist_bins[ind])
            else:
                if ind.size > 1:
                    res[i,j] = np.nanmean(np.array([dist_bins[ind[0]], dist_bins[ind[0]+1]]))
                else:
                    res[i,j] = np.nanmean(np.array([dist_bins[ind], dist_bins[ind+1]]))
    return res
