import numpy as np
from numba import jit, njit
import numba as nb
import illustris_python as il
import insituFuncs as iF
import tracerFuncs as tF
import h5py

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

@njit
def findTracerIDs(star_ids, tracer_ids, parent_ids): 
    #for each element in parent_ids, check whether it's in star_ids (True), otherwise False
    #numpy.nonzero gives indices of elements, where the condition is True
    indices = np.nonzero(isin(parent_ids,star_ids))[0]
    
    return tracer_ids[indices], indices

@njit
def getIndices(search_ids,ids):
    indices = np.nonzero(isin(ids,search_ids))[0]
    return indices

@njit
def sortIDs(parentIDs, tracer_indices):
    #we want to sort all tracers belonging to the same parent particle after each other
    #also sort parent ids: tracer_indices=[(tracers of parent 1),(tracers of parent 2),...]
    #obtain indices that would sort parentIDs
    parent_sorted_indices = np.argsort(parentIDs)
    
    #reshuffle tracer_indices using obtained order from parentIDs
    tracer_sorted_indices = np.ravel(tracer_indices)[parent_sorted_indices]
    
    #return sorted parentIDs as well as reshuffled tracer indices
    return parentIDs[parent_sorted_indices], tracer_sorted_indices

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

@jit(forceobj=True)
def tracersOfSubhalo(basePath,star_ids, start_snap, target_snap):    
    #load tracers at z=0
    tracers = il.snapshot.loadSubset(basePath,start_snap,3)
    tracer_ids = tracers['TracerID'].copy()
    parent_ids = tracers['ParentID'].copy()
    #print('# tracers in total: ',tracer_ids.size)
    
    #load tracers at target snapshot
    target_tracers = il.snapshot.loadSubset(basePath,target_snap,3)
    parent_target_ids = target_tracers['ParentID']
    tracer_target_ids = target_tracers['TracerID']
    
    #find IDs of Tracers belonging to relevant stars
    tracer_search_ids, tracer_indices = findTracerIDs(star_ids,tracer_ids,parent_ids)
    #print('# tracers in subhalo: ',tracer_search_ids.size)
    
    #load all relevant parent particles
    all_gas_ids = il.snapshot.loadSubset(basePath,target_snap,0,['ParticleIDs'])
    all_star_ids = il.snapshot.loadSubset(basePath,target_snap,4,['ParticleIDs'])
    
    #get indices of relevant tracers in array of tracers in target snapshot
    tracer_target_indices = getIndices(tracer_search_ids,tracer_target_ids)
    #print("# tracers in subhalo in target snapshot: ",tracer_target_indices.size)
    
    #sort parents as well as tracers according to sorting of parents
    parent_target_ids, tracer_target_indices = sortIDs(parent_target_ids[tracer_target_indices],tracer_target_indices)
    #print('Are there multiple tracers for some parents? ',not(is_unique(parent_target_ids)))
    #print('Are parent IDs sorted increasingly? ',is_sorted_increasing(parent_target_ids))
    #print('Are the tracer indices aranged in an order that sorts the parent IDs increasingly? ',\
          #is_sorted_increasing(np.ravel(target_tracers['ParentID'].copy())[tracer_target_indices]))
    
    #identify parent types
    gas_indices = getIndices(parent_target_ids,all_gas_ids)
    star_indices = getIndices(parent_target_ids,all_star_ids)
    
    #print("# parent gas particles: ",gas_indices.size)
    #print("# parent star particles: ",star_indices.size)
    
    return all_gas_ids[gas_indices], all_star_ids[star_indices]


@njit(parallel=True)
def insert(arr,index,value):
    assert index<=arr.size
    out = np.zeros(arr.size+1)
    h=0
    for i in range(out.size):
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

@njit
def tracersInSubhalo(StarsInSubOffset, TracersInStarOffset):
    n = StarsInSubOffset.size
    TracersInSubOffset = np.zeros(n)
    for i in nb.prange(n-1):
        TracersInSubOffset[i] = np.sum(TracersInStarOffset[StarsInSubOffset[i]:StarsInSubOffset[i+1]])
    TracersInSubOffset = np.cumsum(TracersInSubOffset)
    return TracersInSubOffset

def createSortedArrays(star_ids,parent_ids,tracer_ids,StarOffsetsInSubs):
    #first determine how many tracers there are for every star
    parent_ids_copy = np.copy(parent_ids)
    parent_ids_copy = np.sort(parent_ids_copy)
    
    num_TracersPerParent = np.searchsorted(parent_ids_copy,star_ids,'right') - np.searchsorted(parent_ids_copy,star_ids,'left')
    parent_ids_offsets = np.cumsum(num_TracersPerParent)
    parent_ids_offsets = np.insert(parent_ids_offsets, 0, 0)
    
#     subIndexTemp = np.zeros(star_ids.size)
#     num_TracersPerSub = np.zeros(StarOffsetsInSubs.size-1)
#     for j in range(StarOffsetsInSubs.size-1):
#         subIndexTemp[StarOffsetsInSubs[j]:StarOffsetsInSubs[j+1]] = j
#         num_TracersPerSub[j] = np.sum(num_TracersPerParent[StarOffsetsInSubs[j]:StarOffsetsInSubs[j+1]])
#     num_TracersPerSub = np.cumsum(num_TracersPerSub)
#     num_TracersPerSub = np.insert(num_TracersPerSub, 0, 0)
    
    #create array, that has the same entries as parent_ids but sorted in a way, that all multiple entries of one ID are located
    #together, e.g. search_parents = [0,0,0,4,4,4,1,2,2,9,9,3,3,3,3,3]
    #similar array, that holds subhalo ids for every star (not necessary)
    search_parents = np.zeros(parent_ids_copy.size)
#    SubIndex = np.zeros(parent_ids_copy.size)
    for i in range(star_ids.size):
        search_parents[parent_ids_offsets[i]:parent_ids_offsets[i+1]] = star_ids[i]
#        SubIndex[parent_ids_offsets[j]:parent_ids_offsets[j+1]] = subIndexTemp[j]
        
    #find the order that would arange parent_ids in the same way search_parents would is aranged
    order = np.argsort(parent_ids)[np.argsort(np.argsort(search_parents))]
    assert( np.array_equal(parent_ids[order], search_parents) )
    
#    del subIndexTemp
    del parent_ids_copy
    
    return tracer_ids[order], parent_ids[order], num_TracersPerParent


def match(ar1, ar2, is_sorted = False):
    """ Returns index arrays i1,i2 of the matching elements between ar1 and ar2. While the elements of ar1 
        must be unique, the elements of ar2 need not be. For every matched element of ar2, the return i1 
        gives the index in ar1 where it can be found. For every matched element of ar1, the return i2 gives 
        the index in ar2 where it can be found. Therefore, ar1[i1] = ar2[i2]. The order of ar2[i2] preserves 
        the order of ar2. Therefore, if all elements of ar2 are in ar1 (e.g. ar1=all TracerIDs in snap, 
        ar2=set of TracerIDs to locate) then ar2[i2] = ar2. The approach is one sort of ar1 followed by 
        bisection search for each element of ar2, therefore O(N_ar1*log(N_ar1) + N_ar2*log(N_ar1)) ~= 
        O(N_ar1*log(N_ar1)) complexity so long as N_ar2 << N_ar1. """
    
    if not is_sorted:
        # need a sorted copy of ar1 to run bisection against
        index = np.argsort(ar1)
        ar1_sorted = ar1[index]
    else:
        ar1_sorted = ar1
        index = np.arange(ar1.shape[0])
    # NlogN search of ar1_sorted for each element in ar2
    ar1_sorted_index = np.searchsorted(ar1_sorted, ar2)

    # undo sort
    ar1_inds = np.take(index, ar1_sorted_index, mode="clip")

    # filter out non-matches
    mask = (ar1[ar1_inds] == ar2)
    ar2_inds = np.where(mask)[0]
    
    #fill non-matches with -1 for later usage
    ar1_inds[np.where(np.logical_not(mask))[0]] = -1
    #ar1_inds = ar1_inds[ar2_inds]

    return ar1_inds, ar2_inds

@njit
def parentIndicesOfAll_slow(parent_ids, all_gas_ids, all_star_ids): 
    #which parent corresponds to which gas/star particles?
    #for this: save index into gas('0') / star('1') subset array
    target_parent_indices = np.zeros((len(parent_ids),2))
    
    for i in nb.prange(len(parent_ids)):
        ind = tF.where_one(parent_ids[i],all_gas_ids)
        if(ind!=-1):
            target_parent_indices[i][0] = int(ind)
            target_parent_indices[i][1] = 0
        else:
            ind2 = tF.where_one(parent_ids[i],all_star_ids)
            target_parent_indices[i][0] = int(ind2)
            target_parent_indices[i][1] = 1
    
    return target_parent_indices

def parentIndicesOfAll(parent_ids, all_gas_ids, all_star_ids): 
    #which parent corresponds to which gas/star particles?
    #for this: save index into gas('0') / star('1') subset array
    target_parent_indices = np.zeros((len(parent_ids),2))
    gas_inds, _ = match(all_gas_ids,parent_ids) #only inds1 relevant
    star_inds, _ = match(all_star_ids,parent_ids) #only inds1 relevant
    
    #gas_inds now contains either the indices of gas parents into the gas subset or -1; same for star_inds
    target_parent_indices[:,0] = gas_inds
    target_parent_indices[:,1] = 0
    target_parent_indices[np.where(gas_inds == -1)[0],1] = 1 #no gas index found => parent is a star
    target_parent_indices[np.where(gas_inds == -1)[0],0] = star_inds[np.where(star_inds!=-1)[0]]
    del gas_inds, star_inds
    return target_parent_indices



def TraceAllStars(basePath,star_ids, start_snap, target_snap, StarsInSubOffset):    
    all_star_ids = il.snapshot.loadSubset(basePath,start_snap,4,['ParticleIDs'])
    #load tracers at z=0
    tracers = il.snapshot.loadSubset(basePath,start_snap,3)
    tracer_ids = tracers['TracerID'].copy()
    parent_ids = tracers['ParentID'].copy()
    
    #load tracers at target snapshot
    target_tracers = il.snapshot.loadSubset(basePath,target_snap,3)
    parent_target_ids = target_tracers['ParentID']
    tracer_target_ids = target_tracers['TracerID']
    
    #load all relevant parent particles at target snapshot
    all_target_gas_ids = il.snapshot.loadSubset(basePath,target_snap,0,['ParticleIDs'])
    all_target_star_ids = il.snapshot.loadSubset(basePath,target_snap,4,['ParticleIDs'])
    
#---------------------------------------------------    
    
    #find IDs of Tracers belonging to relevant stars
    tracer_search_ids, tracer_indices = tF.findTracerIDs(star_ids,tracer_ids,parent_ids)
    
    #rearange all parent and tracer ids
    tracer_search_ids, parent_search_ids, num_tracersInParents =\
    createSortedArrays(star_ids, parent_ids[tracer_indices], tracer_ids[tracer_indices], StarsInSubOffset)
    
    #the number of tracers per subhalo can becomputed from knowing the number of tracers in every star (num_tracersInParent)
    #and the number of stars in every subhalo (StarInSubOffset)
    
    #assert(areEqual(parent_search_ids,parent_ids[tracer_indices]))
    #assert(areEqual(tracer_search_ids,tracer_ids[tracer_indices]))
    
    #get indices of relevant tracers in array of tracers in target snapshot
    tracer_target_indices = tF.getIndices(tracer_search_ids,tracer_target_ids)
    
    target_search_tracer_ids = tracer_target_ids[tracer_target_indices].copy()
    target_search_parent_ids = parent_target_ids[tracer_target_indices].copy()
    
    original_order = np.argsort(target_search_tracer_ids)[np.argsort(np.argsort(tracer_search_ids))]
    
    #sort parents as well as tracers according to sorting of parents
    target_search_tracer_ids = target_search_tracer_ids[original_order]
    target_search_parent_ids = target_search_parent_ids[original_order]
    
    assert(np.array_equal(target_search_tracer_ids,tracer_search_ids))
    
    target_parent_indices = parentIndicesOfAll(parent_ids = target_search_parent_ids, all_gas_ids = all_target_gas_ids,\
                                               all_star_ids = all_target_star_ids)
    
    
    return target_parent_indices, num_tracersInParents

#define function that saves results from TraceAllStars
def TraceBackAllInsituStars(basePath,start_snap,target_snap):
    #load all star ids from a specific galaxy
    star_ids = il.snapshot.loadSubset(basePath,start_snap,'stars',fields=['ParticleIDs'])
    sub_coms = il.groupcat.loadSubhalos(basePath,target_snap,fields=['SubhaloCM'])

    #determine all stars from that galaxy that were formed insitu
    insitu = iF.is_insitu(basePath,np.arange(star_ids.size),start_snap)
    insitu_star_indices = np.nonzero(insitu)[0]
    
    #load postprocessing file that contains information about offsets of stars in subhalos
    f = h5py.File(basePath[:-6] + 'postprocessing/offsets/offsets_0' + str(start_snap) + '.hdf5','r')
    starsInSubOffset = f['Subhalo/SnapByType'][:,4]
    f.close()
    
    #calculate _inSitu_ star offsets
    check = h5py.File(basePath[:-6] + 'postprocessing/StellarAssembly/stars_0' + str(start_snap) + '.hdf5','r')
    insitu = np.asarray(check['InSitu'][:]==1)
    check.close()
    insituStarsInSubOffset = np.zeros(starsInSubOffset.shape[0])
    for i in range(1,starsInSubOffset.shape[0]):
        star_indices = np.arange(starsInSubOffset[i-1],starsInSubOffset[i])
        insitu_indices = insitu[star_indices]
        insituStarsInSubOffset[i] = len(np.nonzero(insitu_indices)[0])
    insituStarsInSubOffset = np.cumsum(insituStarsInSubOffset)
    
    #run function
    parent_indices, tracersInSubOffset = TraceAllStars(basePath,star_ids[insitu_star_indices],\
                                                       start_snap,target_snap,insituStarsInSubOffset)
    
    redshift = il.groupcat.loadHeader(basePath,target_snap)['Redshift']
    
    #save results in hdf5 file
    sim = basePath[32:39]
    result = h5py.File('files/'+sim+'/parent_indices_redshift_{:.1f}.hdf5'.format(redshift),'w')
    dset = result.create_dataset("parent_indices", parent_indices.shape, dtype=float)
    dset[:] = parent_indices
    dset2 = result.create_dataset('tracers_in_parents_offset',tracersInSubOffset.shape, dtype=float)
    dset2[:] = tracersInSubOffset
    result.close()
    return

def AllTracerProfile(basePath,start_snap,target_snap):
    header = il.groupcat.loadHeader(basePath,target_snap)
    redshift = header['Redshift']
    h_const = header['HubbleParam']
    boxSize = header['BoxSize']
    
    parent_indices = h5py.File('files/'+basePath[32:39]+'/parent_indices_redshift_{:.1f}.hdf5'.format(redshift),'r')
    sub_positions = h5py.File('files/'basePath[32:39]'/SubhaloPosAtAllSnaps_v2-Copy1_extrapolated.hdf5','r') 
    #possibly the position at that snapshot had to be extrapolated
    
    sub_pos_at_target_snap = sub_positions['SubhaloPos'][:,:,:]
    num_subs = sub_pos_at_target_snap.shape[0]

    parent_indices_data = parent_indices['parent_indices'][:,:]
    tracers_in_parent_offset = parent_indices['tracers_in_parents_offset'][:]
    
    all_gas_pos = il.snapshot.loadSubset(basePath,target_snap,'gas',fields=['Coordinates'])

    f = h5py.File(basePath[:-6] + 'postprocessing/offsets/offsets_0' + str(start_snap) + '.hdf5','r')
    starsInSubOffset = f['Subhalo/SnapByType'][:,4]
    f.close()
    
    check = h5py.File(basePath[:-6] + 'postprocessing/StellarAssembly/stars_0' + str(start_snap) + '.hdf5','r')
    insitu = np.asarray(check['InSitu'][:]==1)
    check.close()
    insituStarsInSubOffset = np.zeros(starsInSubOffset.shape[0])
    for i in range(1,starsInSubOffset.shape[0]):
        star_indices = np.arange(starsInSubOffset[i-1],starsInSubOffset[i])
        insitu_indices = insitu[star_indices]
        insituStarsInSubOffset[i] = len(np.nonzero(insitu_indices)[0])
    insituStarsInSubOffset = np.cumsum(insituStarsInSubOffset)
    
    #there might be more tracers -> parents in one galaxy at higher redshifts than insitu stars at redshift 0
    final_offsets = tF.tracersInSubhalo(insituStarsInSubOffset,tracers_in_parent_offset)
    
    rad_profile = np.zeros(1)
    for i in range(1,num_subs):
        parent_indices_of_sub = parent_indices_data[int(final_offsets[i-1]):int(final_offsets[i]),:]
        gas_parent_indices = parent_indices_of_sub[np.where(parent_indices_of_sub[:,1]==0)[0],0]
        gas_pos = all_gas_pos[gas_parent_indices.astype('int')]
    
        subhalo_position = sub_pos_at_target_snap[i-1,start_snap-target_snap,:]
    
        rad_dist = np.ones(gas_pos.shape[0])
        for j in range(gas_pos.shape[0]):
            rad_dist[j] = iF.dist(subhalo_position,gas_pos[j],boxSize)
        rad_profile = np.concatenate((rad_profile,rad_dist))
        
    print(rad_profile.shape)
    bins, num = iF.binData(rad_profile[np.where(rad_profile<boxSize)[0]],100)

    parent_indices.close()
    sub_positions.close()
    return bins, num