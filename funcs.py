import numba as nb
import numpy as np
from numba import jit, njit

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

@njit(parallel=True)
def insert(arr,index,value):
    assert index<=arr.size
    out = np.zeros(arr.size+1)
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
    # Create a hash table to count number of instances
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
