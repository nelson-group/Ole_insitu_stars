import numpy as np
from numba import jit, njit
import numba as nb
import illustris_python as il
import tracerFuncs as tF
import funcs
import h5py
<<<<<<< HEAD
import time
=======

>>>>>>> de57ecd0d8a38fa9869bb489726e23ead107bc54

@njit
def findTracerIDs(star_ids, tracer_ids, parent_ids): 
    #for each element in parent_ids, check whether it's in star_ids (True), otherwise False
    #numpy.nonzero gives indices of elements, where the condition is True
    indices = np.nonzero(funcs.isin(parent_ids,star_ids))[0]
    
    return tracer_ids[indices], indices

@njit
def getIndices(search_ids,ids):
<<<<<<< HEAD
    """Returns the indices into ids of all elements of search_ids that are also elements of ids."""
    indices = np.nonzero(funcs.isin(ids,search_ids))[0]
    return indices

# @njit
# def sortIDs(parentIDs, tracer_indices):
#     #we want to sort all tracers belonging to the same parent particle after each other
#     #also sort parent ids: tracer_indices=[(tracers of parent 1),(tracers of parent 2),...]
#     #obtain indices that would sort parentIDs
#     parent_sorted_indices = np.argsort(parentIDs)
    
#     #reshuffle tracer_indices using obtained order from parentIDs
#     tracer_sorted_indices = np.ravel(tracer_indices)[parent_sorted_indices]
    
#     #return sorted parentIDs as well as reshuffled tracer indices
#     return parentIDs[parent_sorted_indices], tracer_sorted_indices


# def tracersOfSubhalo(basePath,star_ids, start_snap, target_snap):    
#     #load tracers at z=0
#     tracers = il.snapshot.loadSubset(basePath,start_snap,3)
#     tracer_ids = tracers['TracerID'].copy()
#     parent_ids = tracers['ParentID'].copy()
#     del tracers
#     #print('# tracers in total: ',tracer_ids.size)
    
#     #load tracers at target snapshot
#     target_tracers = il.snapshot.loadSubset(basePath,target_snap,3)
#     parent_target_ids = target_tracers['ParentID'].copy()
#     tracer_target_ids = target_tracers['TracerID'].copy()
#     del target_tracers
    
#     #find IDs of Tracers belonging to relevant stars
#     tracer_search_ids, tracer_indices = findTracerIDs(star_ids,tracer_ids,parent_ids)
    
#     #load all relevant parent particles
#     all_gas_ids = il.snapshot.loadSubset(basePath,target_snap,0,['ParticleIDs'])
#     all_star_ids = il.snapshot.loadSubset(basePath,target_snap,4,['ParticleIDs'])
    
#     #get indices of relevant tracers in array of tracers in target snapshot
#     tracer_target_indices = getIndices(tracer_search_ids,tracer_target_ids)
    
#     #sort parents as well as tracers according to sorting of parents
#     parent_target_ids, tracer_target_indices = sortIDs(parent_target_ids[tracer_target_indices],tracer_target_indices)
    
#     #identify parent types
#     gas_indices = getIndices(parent_target_ids,all_gas_ids)
#     star_indices = getIndices(parent_target_ids,all_star_ids)
    
#     return all_gas_ids[gas_indices], all_star_ids[star_indices]


@jit(nopython = True)
=======
    indices = np.nonzero(funcs.isin(ids,search_ids))[0]
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
    
    #load all relevant parent particles
    all_gas_ids = il.snapshot.loadSubset(basePath,target_snap,0,['ParticleIDs'])
    all_star_ids = il.snapshot.loadSubset(basePath,target_snap,4,['ParticleIDs'])
    
    #get indices of relevant tracers in array of tracers in target snapshot
    tracer_target_indices = getIndices(tracer_search_ids,tracer_target_ids)
    
    #sort parents as well as tracers according to sorting of parents
    parent_target_ids, tracer_target_indices = sortIDs(parent_target_ids[tracer_target_indices],tracer_target_indices)
    
    #identify parent types
    gas_indices = getIndices(parent_target_ids,all_gas_ids)
    star_indices = getIndices(parent_target_ids,all_star_ids)
    
    return all_gas_ids[gas_indices], all_star_ids[star_indices]


@njit
>>>>>>> de57ecd0d8a38fa9869bb489726e23ead107bc54
def tracersInSubhalo(StarsInSubOffset, TracersInStarOffset):
    n = StarsInSubOffset.shape[0]
    TracersInSubOffset = np.zeros(n)
    for i in nb.prange(n-1):
        TracersInSubOffset[i] = np.sum(TracersInStarOffset[StarsInSubOffset[i]:StarsInSubOffset[i+1]])
    TracersInSubOffset = np.cumsum(TracersInSubOffset)
    return TracersInSubOffset

<<<<<<< HEAD
#@jit(nopython = True, parallel = True)
def createSortedArrays(star_ids, parent_ids, tracer_ids, StarOffsetsInSubs):
=======
def createSortedArrays(star_ids,parent_ids,tracer_ids,StarOffsetsInSubs):
>>>>>>> de57ecd0d8a38fa9869bb489726e23ead107bc54
    #first determine how many tracers there are for every star
    parent_ids_copy = np.copy(parent_ids)
    parent_ids_copy = np.sort(parent_ids_copy)
    
<<<<<<< HEAD
    num_TracersPerParent = funcs.searchsorted_nb_right(parent_ids_copy,star_ids) - funcs.searchsorted_nb_left(parent_ids_copy,star_ids)
    parent_ids_offsets = np.cumsum(num_TracersPerParent)
    parent_ids_offsets = np.insert(parent_ids_offsets,0,0)
=======
    num_TracersPerParent = np.searchsorted(parent_ids_copy,star_ids,'right') - np.searchsorted(parent_ids_copy,star_ids,'left')
    parent_ids_offsets = np.cumsum(num_TracersPerParent)
    parent_ids_offsets = np.insert(parent_ids_offsets, 0, 0)
>>>>>>> de57ecd0d8a38fa9869bb489726e23ead107bc54
    
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
<<<<<<< HEAD
#    assert( np.array_equal(parent_ids[order], search_parents) )
=======
    assert( np.array_equal(parent_ids[order], search_parents) )
>>>>>>> de57ecd0d8a38fa9869bb489726e23ead107bc54
    
#    del subIndexTemp
    del parent_ids_copy
    
    return tracer_ids[order], parent_ids[order], num_TracersPerParent

<<<<<<< HEAD
# @jit(nopython = True, parallel = True)
=======

>>>>>>> de57ecd0d8a38fa9869bb489726e23ead107bc54
def match(ar1, ar2, is_sorted = False):
    """ Returns index arrays i1,i2 of the matching elements between ar1 and ar2. While the elements of ar1 
        must be unique, the elements of ar2 need not be. For every matched element of ar2, the return i1 
        gives the index in ar1 where it can be found. For every matched element of ar1, the return i2 gives 
        the index in ar2 where it can be found. Therefore, ar1[i1] = ar2[i2]. The order of ar2[i2] preserves 
        the order of ar2. Therefore, if all elements of ar2 are in ar1 (e.g. ar1=all TracerIDs in snap, 
        ar2=set of TracerIDs to locate) then ar2[i2] = ar2. The approach is one sort of ar1 followed by 
        bisection search for each element of ar2, therefore O(N_ar1*log(N_ar1) + N_ar2*log(N_ar1)) ~= 
        O(N_ar1*log(N_ar1)) complexity so long as N_ar2 << N_ar1. """
    
<<<<<<< HEAD
    if type(ar1) == dict:        
        return -1, np.array([])
    
=======
>>>>>>> de57ecd0d8a38fa9869bb489726e23ead107bc54
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
<<<<<<< HEAD
#     ar1_inds = funcs.take_nb_clip(index, ar1_sorted_index)
=======
>>>>>>> de57ecd0d8a38fa9869bb489726e23ead107bc54

    # filter out non-matches
    mask = (ar1[ar1_inds] == ar2)
    ar2_inds = np.where(mask)[0]
    
<<<<<<< HEAD
    #out = 'last true' if mask[-1] else 'last false'
=======
    out = 'last true' if mask[-1] else 'last false'
>>>>>>> de57ecd0d8a38fa9869bb489726e23ead107bc54
    #print(out)
    
    #fill non-matches with -1 for later usage
    ar1_inds[np.where(np.logical_not(mask))[0]] = -1 #shape of gas_inds doen't match
    
    #ar1_inds = ar1_inds[ar2_inds]
    return ar1_inds, ar2_inds

<<<<<<< HEAD
# @jit(nopython = True, parallel = True)
def match_general(ar1, ar2, is_sorted = False):
    """ Returns index arrays i1,i2 of the matching elements between ar1 and ar2. While the elements of ar1 
        must be unique, the elements of ar2 need not be. For every matched element of ar2, the return i1 
        gives the index in ar1 where it can be found. For every matched element of ar1, the return i2 gives 
        the index in ar2 where it can be found. Therefore, ar1[i1] = ar2[i2]. The order of ar2[i2] preserves 
        the order of ar2. Therefore, if all elements of ar2 are in ar1 (e.g. ar1=all TracerIDs in snap, 
        ar2=set of TracerIDs to locate) then ar2[i2] = ar2. The approach is one sort of ar1 followed by 
        bisection search for each element of ar2, therefore O(N_ar1*log(N_ar1) + N_ar2*log(N_ar1)) ~= 
        O(N_ar1*log(N_ar1)) complexity so long as N_ar2 << N_ar1. """
    
    if type(ar1) == dict:        
        return -1, np.array([])
    
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
#     ar1_inds = funcs.take_nb_clip(index, ar1_sorted_index)

    # filter out non-matches
    mask = (ar1[ar1_inds] == ar2)
    ar2_inds = np.where(mask)[0]
    ar1_inds = ar1_inds[ar2_inds]
    
    #fill non-matches with -1 for later usage
#     ar1_inds[np.where(np.logical_not(mask))[0]] = -1 #shape of gas_inds doen't match
    
    return ar1_inds, ar2_inds

# @jit(nopython = True, parallel = True)
=======
@njit
def parentIndicesOfAll_slow(parent_ids, all_gas_ids, all_star_ids): #N*N
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

>>>>>>> de57ecd0d8a38fa9869bb489726e23ead107bc54
def parentIndicesOfAll(parent_ids, all_gas_ids, all_star_ids): #NlogN
    #which parent corresponds to which gas/star particles?
    #for this: save index into gas('0') / star('1') subset array
    target_parent_indices = np.zeros((len(parent_ids),2))
<<<<<<< HEAD
    gas_inds, _ = tF.match(all_gas_ids, parent_ids) #only inds1 relevant
    print('gas indices done')
    
    if all_star_ids.shape[0] > 0:
        star_inds, _ = tF.match(all_star_ids, parent_ids) #only inds1 relevant
    else:
        star_inds = np.array([])
    print('star indices done')
    #check for inconsistencies
    #print(np.where(gas_inds == -1)[0].shape[0])
    #print(np.where(star_inds != -1)[0].shape[0])
    #assert(np.where(gas_inds == -1)[0].shape[0] == np.where(star_inds != -1)[0].shape[0])
=======
    gas_inds, _ = match(all_gas_ids, parent_ids) #only inds1 relevant
    star_inds, _ = match(all_star_ids, parent_ids) #only inds1 relevant
>>>>>>> de57ecd0d8a38fa9869bb489726e23ead107bc54
    
    #gas_inds now contains either the indices of gas parents into the gas subset or -1; same for star_inds
    target_parent_indices[:,0] = gas_inds
    target_parent_indices[:,1] = 0
<<<<<<< HEAD
    mask = np.where(gas_inds == -1)[0]
    target_parent_indices[mask,1] = 1 #no gas index found => parent is a star
    if type(star_inds) == np.ndarray:
        target_parent_indices[mask,0] = star_inds[np.where(star_inds != -1)[0]][:mask.shape[0]]
    elif type(star_inds) == int:
        target_parent_indices[mask,0] = star_inds
    del gas_inds, star_inds
    print('filling done')
=======
    target_parent_indices[np.where(gas_inds == -1)[0],1] = 1 #no gas index found => parent is a star
    target_parent_indices[np.where(gas_inds == -1)[0],0] = star_inds[np.where(star_inds != -1)[0]]
    del gas_inds, star_inds
>>>>>>> de57ecd0d8a38fa9869bb489726e23ead107bc54
    return target_parent_indices



def TraceAllStars(basePath,star_ids, start_snap, target_snap, StarsInSubOffset):    
    #all_star_ids = il.snapshot.loadSubset(basePath,start_snap,4,['ParticleIDs'])
    #load tracers at z=0
<<<<<<< HEAD
    start = time.time()
    tracer_ids = il.snapshot.loadSubset(basePath,start_snap,3, fields = ['TracerID'])
    parent_ids = il.snapshot.loadSubset(basePath,start_snap,3, fields = ['ParentID'])
    
    load = time.time()
    print('time for tracer loading: ',load-start)
#---------------------------------------------------    
    
    #find IDs of Tracers belonging to relevant stars
    _, tracer_indices = tF.findTracerIDs(star_ids,tracer_ids,parent_ids)
    
    find_tracers = time.time()
    print('time for finding tracers: ',find_tracers-load, flush = True)
    
    #rearange all parent and tracer ids
    tracer_search_ids, _, numTracersInParents =\
    createSortedArrays(star_ids, parent_ids[tracer_indices], tracer_ids[tracer_indices], StarsInSubOffset)
    
    create_arrays = time.time()
    print('time for creating sorted arrays: ',create_arrays-find_tracers, flush = True)
    del parent_ids, tracer_ids, StarsInSubOffset, star_ids
=======
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
    tracer_search_ids, parent_search_ids, numTracersInParents =\
    createSortedArrays(star_ids, parent_ids[tracer_indices], tracer_ids[tracer_indices], StarsInSubOffset)
    
>>>>>>> de57ecd0d8a38fa9869bb489726e23ead107bc54
    #the number of tracers per subhalo can becomputed from knowing the number of tracers in every star (numTracersInParent)
    #and the number of stars in every subhalo (StarInSubOffset)
    
    #assert(areEqual(parent_search_ids,parent_ids[tracer_indices]))
    #assert(areEqual(tracer_search_ids,tracer_ids[tracer_indices]))
    
<<<<<<< HEAD
    #load tracers at target snapshot
    tracer_target_ids = il.snapshot.loadSubset(basePath,target_snap,3, fields = ['TracerID'])
    parent_target_ids = il.snapshot.loadSubset(basePath,target_snap,3, fields = ['ParentID'])
    
    after_load2 = time.time()
    
=======
>>>>>>> de57ecd0d8a38fa9869bb489726e23ead107bc54
    #get indices of relevant tracers in array of tracers in target snapshot
    tracer_target_indices = tF.getIndices(tracer_search_ids,tracer_target_ids)
    
    target_tracer_search_ids = tracer_target_ids[tracer_target_indices].copy()
    target_parent_search_ids = parent_target_ids[tracer_target_indices].copy()
    
<<<<<<< HEAD
    del tracer_target_ids, parent_target_ids
    
    original_order = np.argsort(target_tracer_search_ids)[np.argsort(np.argsort(tracer_search_ids))]
    
    del tracer_search_ids, target_tracer_search_ids
    
    sort = time.time()
    print('time for getting indices and sorting: ',sort - after_load2, flush = True)
    
    #sort parents as well as tracers according to sorting of parents at start snap
    #target_tracer_search_ids = target_tracer_search_ids[original_order]
    target_parent_search_ids = target_parent_search_ids[original_order]
    
    del original_order
    
    #assert(np.array_equal(target_tracer_search_ids,tracer_search_ids))
    
    #load all relevant parent particles at target snapshot
    all_target_gas_ids = il.snapshot.loadSubset(basePath,target_snap,0,['ParticleIDs'])
    all_target_star_ids = il.snapshot.loadSubset(basePath,target_snap,4,['ParticleIDs'])
    
    if type(all_target_star_ids) == dict:
        all_target_star_ids = np.array([])
    
    after_load3 = time.time()
=======
    original_order = np.argsort(target_tracer_search_ids)[np.argsort(np.argsort(tracer_search_ids))]
    
    #sort parents as well as tracers according to sorting of parents at start snap
    target_tracer_search_ids = target_tracer_search_ids[original_order]
    target_parent_search_ids = target_parent_search_ids[original_order]
    
    assert(np.array_equal(target_tracer_search_ids,tracer_search_ids))
>>>>>>> de57ecd0d8a38fa9869bb489726e23ead107bc54
    
    target_parent_indices = parentIndicesOfAll(parent_ids = target_parent_search_ids, all_gas_ids = all_target_gas_ids,\
                                               all_star_ids = all_target_star_ids)
    
<<<<<<< HEAD
    end = time.time()
    print('time for finding indices into subset arrays: ',end-after_load3, flush = True)
    print('total time: ',end-start, flush = True)
    return target_parent_indices, numTracersInParents

#define function that saves results from TraceAllStars now for every single snapshot
def TraceBackAllInsituStars_allSnaps(basePath,start_snap):
    #load all star ids from a specific galaxy
    star_ids = il.snapshot.loadSubset(basePath,start_snap,'stars',fields=['ParticleIDs'])

    #determine all stars that were formed insitu
    insitu = funcs.is_insitu(basePath,start_snap)
    insitu = np.asarray(insitu == 1)
    insitu_star_indices = np.nonzero(insitu)[0]
    
    insituStarsInSubOffset = tF.insituStarsInSubOffset(basePath, start_snap)
    
    sim = basePath[32:39]
    
    result = h5py.File('files/' + sim + '/all_parent_indices_test.hdf5','w')
    #result = h5py.File('../../tmp/all_parent_indices.hdf5','w')    
    
    #run function for every snapshot
    for target_snap in np.arange(99,1,-1):
        parent_indices, numTracersInParents = tF.TraceAllStars(basePath,star_ids[insitu_star_indices],\
                                                       start_snap,target_snap,insituStarsInSubOffset)  
        #save results in hdf5 file

        grp = result.create_group(f'snap_0{target_snap}')
        dset = grp.create_dataset("parent_indices", parent_indices.shape, dtype=float)
        dset[:] = parent_indices
        dset2 = grp.create_dataset('numTracersInParents',numTracersInParents.shape, dtype=float)
        dset2[:] = numTracersInParents
        print(target_snap, 'done',end = '; ', flush=True)
    result.close()
    return

=======
    
    return target_parent_indices, numTracersInParents

#define function that saves results from TraceAllStars
def TraceBackAllInsituStars(basePath,start_snap,target_snap):
    #load all star ids from a specific galaxy
    star_ids = il.snapshot.loadSubset(basePath,start_snap,'stars',fields=['ParticleIDs'])

    #determine all stars from that galaxy that were formed insitu
    check = h5py.File(basePath[:-6] + 'postprocessing/StellarAssembly/stars_0' + str(start_snap) + '.hdf5','r')
    insitu = check['InSitu'][:] #1 if star is formed insitu and 0 otherwise
    check.close()
    insitu_star_indices = np.nonzero(insitu)[0]
    
    insituStarsInSubOffset_start_snap = tF.insituStarsInSubOffset(basePath, start_snap)
    
    #run function
    parent_indices, tracersInParentsOffset = TraceAllStars(basePath,star_ids[insitu_star_indices],\
                                                       start_snap,target_snap,insituStarsInSubOffset_start_snap)
    
    redshift = il.groupcat.loadHeader(basePath,target_snap)['Redshift']
    
    #save results in hdf5 file
    sim = basePath[32:39]
    result = h5py.File('files/'+sim+'/parent_indices_redshift_{:.1f}.hdf5'.format(redshift),'w')
    dset = result.create_dataset("parent_indices", parent_indices.shape, dtype=float)
    dset[:] = parent_indices
    dset2 = result.create_dataset('tracers_in_parents_offset',tracersInParentsOffset.shape, dtype=float)
    dset2[:] = tracersInParentsOffset
    result.close()
    return

def AllTracerProfile(basePath,start_snap,target_snap):
    header = il.groupcat.loadHeader(basePath,target_snap)
    redshift = header['Redshift']
    h_const = header['HubbleParam']
    boxSize = header['BoxSize']
    
    parent_indices = h5py.File('files/'+basePath[32:39]+'/parent_indices_redshift_{:.1f}.hdf5'.format(redshift),'r')
    sub_positions = h5py.File('files/'+basePath[32:39]+'/SubhaloPosAtAllSnaps_v2-Copy1_extrapolated.hdf5','r') 
    #possibly the position at that snapshot had to be extrapolated
    
    sub_pos_at_target_snap = sub_positions['SubhaloPos'][:,:,:]
    num_subs = sub_pos_at_target_snap.shape[0]

    parent_indices_data = parent_indices['parent_indices'][:,:]
    tracers_in_parent_offset = parent_indices['tracers_in_parents_offset'][:]
    
    all_gas_pos = il.snapshot.loadSubset(basePath,target_snap,'gas',fields=['Coordinates'])

    insituStarsInSubOffset_start_snap = tF.insituStarsInSubOffset(basePath, start_snap)
    
    #there might be more tracers -> parents in one galaxy at higher redshifts than insitu stars at redshift 0
    final_offsets = tF.tracersInSubhalo(insituStarsInSubOffset_start_snap, tracers_in_parent_offset)
    
    rad_profile = np.zeros(1)
    for i in range(1,num_subs):
        parent_indices_of_sub = parent_indices_data[int(final_offsets[i-1]):int(final_offsets[i]),:]
        gas_parent_indices = parent_indices_of_sub[np.where(parent_indices_of_sub[:,1]==0)[0],0]
        gas_pos = all_gas_pos[gas_parent_indices.astype('int')]
    
        subhalo_position = sub_pos_at_target_snap[i-1,start_snap-target_snap,:]
    
        rad_dist = np.ones(gas_pos.shape[0])
        for j in range(gas_pos.shape[0]):
            rad_dist[j] = funcs.dist(subhalo_position,gas_pos[j],boxSize)
        rad_profile = np.concatenate((rad_profile,rad_dist))
        
    print(rad_profile.shape)
    bins, num = iF.binData(rad_profile[np.where(rad_profile<boxSize)[0]],100)

    parent_indices.close()
    sub_positions.close()
    return bins, num
>>>>>>> de57ecd0d8a38fa9869bb489726e23ead107bc54

def insituStarsInSubOffset(basePath, snap):
    """compute for an array of insitu stars, how many of them are in each subhalo
    """
    if snap < 10:
        str_snap = f'0{snap}'
    else:
        str_snap = str(snap)
    g = h5py.File(basePath[:-6] + 'postprocessing/offsets/offsets_0' + str_snap + '.hdf5','r')
    starsInSubOffset = g['Subhalo/SnapByType'][:,4]
    g.close()
    numStarsInSubs = il.groupcat.loadSubhalos(basePath, snap, fields = ['SubhaloLenType'])[:,4]
    
    check = h5py.File(basePath[:-6] + 'postprocessing/StellarAssembly/stars_0' + str_snap + '.hdf5','r')
<<<<<<< HEAD
    insitu = check['InSitu'][:] #1 if star is formed insitu, 0 if it was formed ex-situ and -1 otherwise (fuzz)
    insitu = np.asarray(insitu == 1)
    check.close()
    
    insituStarsInSubOffset = compute_offsets(starsInSubOffset, numStarsInSubs, insitu)
#     insituStarsInSubOffset = np.zeros(starsInSubOffset.shape[0])
#     for i in nb.prange(1,starsInSubOffset.shape[0]):
#         star_indices = np.arange(starsInSubOffset[i-1],starsInSubOffset[i-1] +\
#                                  numStarsInSubs[i-1])
#         insitu_indices = insitu[star_indices]
#         insituStarsInSubOffset[i] = len(np.nonzero(insitu_indices)[0])
    
#     insituStarsInSubOffset = np.cumsum(insituStarsInSubOffset)
    return insituStarsInSubOffset

def exsituStarsInSubOffset(basePath, snap):
    """computes the number of exsitu stars in each subhalo, outputs the cumulative sum
    """
    g = h5py.File(basePath[:-6] + 'postprocessing/offsets/offsets_' + str(snap).zfill(3) + '.hdf5','r')
    starsInSubOffset = g['Subhalo/SnapByType'][:,4]
    g.close()
    numStarsInSubs = il.groupcat.loadSubhalos(basePath, snap, fields = ['SubhaloLenType'])[:,4]
    
    check = h5py.File(basePath[:-6] + 'postprocessing/StellarAssembly/stars_' + str(snap).zfill(3) + '.hdf5','r')
    exsitu = check['InSitu'][:] #1 if star is formed insitu, 0 if it was formed ex-situ and -1 otherwise (fuzz)
    exsitu = np.asarray(exsitu == 0) # array of True when star formed ex-situ and False otherwise
    check.close()
    
    exsituStarsInSubOffset = compute_offsets(starsInSubOffset, numStarsInSubs, exsitu)
    return exsituStarsInSubOffset

@jit(nopython = True, parallel = True)
def compute_offsets(starsInSubOffset, numStarsInSubs, insitu):
    insituStarsInSubOffset = np.zeros(starsInSubOffset.shape[0], dtype = np.int32)
    for i in nb.prange(1,starsInSubOffset.shape[0]):
=======
    insitu = check['InSitu'][:] #1 if star is formed insitu and 0 otherwise
    check.close()
    
    insituStarsInSubOffset = np.zeros(starsInSubOffset.shape[0])
    for i in range(1,starsInSubOffset.shape[0]):
>>>>>>> de57ecd0d8a38fa9869bb489726e23ead107bc54
        star_indices = np.arange(starsInSubOffset[i-1],starsInSubOffset[i-1] +\
                                 numStarsInSubs[i-1])
        insitu_indices = insitu[star_indices]
        insituStarsInSubOffset[i] = len(np.nonzero(insitu_indices)[0])
    
    insituStarsInSubOffset = np.cumsum(insituStarsInSubOffset)
<<<<<<< HEAD
    return insituStarsInSubOffset


@jit(nopython=True, parallel = True)
def formation_snapshot(hmr, snaps):
    form_snap = np.zeros(hmr.shape[1])
    for i in nb.prange(hmr.shape[1]):
        ind = np.where(hmr[:,i] == 1)[0]
        if ind.size > 0:
            if ind.size == 1:
                form_snap[i] = snaps[ind[0]]
            else:
                form_snap[i] = np.nanmean(snaps[ind.flatten()])
            continue
        r_dist = 1 - hmr[:,i]
        if np.all(r_dist > 1):
            form_snap[i] = -1
            continue
        ind = np.where(r_dist[1:] * r_dist[:-1] < 0)[0]
        if ind.size == 0: #if there's no match at all, put snapshot 0 => most likely the hmr hasn't reached one yet
            form_snap[i] = snaps[-1]
            continue

        m = (hmr[ind[0]+1,i] - hmr[ind[0],i])/(snaps[ind[0]+1] - snaps[ind[0]])
        form_snap[i] = (1 - hmr[ind[0],i])/m + snaps[ind[0]]
    return form_snap

@jit(nopython = True, parallel = True)
def hmrs(res, cumsum_profiles, dist_bins):
    for i in nb.prange(cumsum_profiles.shape[0]):
        for j in nb.prange(cumsum_profiles.shape[1]): #for all remaining centrals
            if np.all(cumsum_profiles[i,j,:] == -1):
                res[i,j] = -1
                continue
            ind, next_too = funcs.data_intersect_value(cumsum_profiles[i,j,:],0.5)
            if not next_too:
                res[i,j] = np.mean(dist_bins[ind])
            else:
                if ind.size > 1:
                    res[i,j] = np.nanmean(np.array([dist_bins[ind[0]], dist_bins[ind[0]+1]]))
                else:
                    a = np.concatenate((dist_bins[ind].flatten(), dist_bins[ind+1].flatten()))
                    res[i,j] = np.nanmean(a)
    return res

def get_halfmass_radii(run, max_dist, snaps):
    """Computes the halfmass radii and estimates the formation snapshots (z at which hmr == R_vir)
    for all central galaxies."""

    file = '/vera/ptmp/gc/olwitt/TNG50-' + run + f'/cumulative_radial_profile/rad_prof_tracer_frac_{max_dist}_all_99.hdf5'
    #assert isfile(file), 'filename wrong'
    f = h5py.File(file,'r')
    profiles = f['cumulative_profiles'][:,:]
    dist_bins = f['distance_bins'][:]
    extrapolated_sub_ids = f['which_galaxy_ids'][:]
#     sub_medians = f['subhalo_median_distances'][:]
    f.close()
    
    cumsum_profiles = np.empty((snaps.shape[0],profiles.shape[0],profiles.shape[1]), dtype = profiles.dtype)
#     hmr_tracers = np.empty((snaps.shape[0],sub_medians.shape[0]),dtype = sub_medians.dtype)
    for i in range(snaps.size):
        f = h5py.File('/vera/ptmp/gc/olwitt/TNG50-' + run +\
                      f'/cumulative_radial_profile/rad_prof_tracer_frac_{max_dist}_all_{snaps[i]}.hdf5','r')
        profiles = f['cumulative_profiles'][:,:]
        extrapolated_sub_ids = f['which_galaxy_ids'][:]
#         sub_medians = f['subhalo_median_distances'][:]
        f.close() 
        if profiles.shape[0] != cumsum_profiles.shape[1]:
            #if there are less profiles in the snapshot file than at snap99: add -1 arrays so the shapes match again
            diff = cumsum_profiles.shape[1] - profiles.shape[0]
            counter = 0
            for j in range(cumsum_profiles.shape[1]):
                if extrapolated_sub_ids[j] == -1:
                    cumsum_profiles[i,j,:] = np.full(cumsum_profiles.shape[2],-1)
#                     hmr_tracers[i,j] = np.nan
                    diff -= 1
                else:
                    cumsum_profiles[i,j,:] = profiles[counter,:]
#                     hmr_tracers[i,j] = sub_medians[counter]
                    counter += 1
        else:
            cumsum_profiles[i,:,:] = profiles
#             hmr_tracers[i,:] = sub_medians
        print('snap ',snaps[i],' loaded;')
    del profiles
    
    print('loading complete...')
    
    hmr = np.zeros((cumsum_profiles.shape[0],cumsum_profiles.shape[1]))
    
    hmr = hmrs(hmr,cumsum_profiles,dist_bins)
    
    print('halfmass radii computation complete...')
    
    #now compute formation redshifts, i.e. the interpolated redshift at which the halfmass radius is equal to R_vir
    form_snap = formation_snapshot(hmr, snaps)
    
    print('formation snapshot computation complete...')
    
    f = h5py.File('files/'+basePath[32:39]+f'/hmr_form_snap_{max_dist}.hdf5','w')
    ds = f.create_dataset('halfmass_radii',data = hmr)
#     ds1 = f.create_dataset('halfmass_radii_from_tracers', data = hmr_tracers)
    ds2 = f.create_dataset('formation_snapshots',data = form_snap)
    f.close()
    
    
    return hmr, form_snap

#@jit(nopython = True, parallel = True)
def DM_in_2_shmr(DM_coords, sub_pos, dmInSubOffset, numDMInSubs, shmr, cut, boxSize):
    which_particles = np.zeros(DM_coords.shape[0], dtype = np.ubyte)
    inside_offsets = np.zeros(sub_pos.shape[0], dtype = int)
    
    for i in range(sub_pos.shape[0]):
        sub_indices = np.arange(dmInSubOffset[i],dmInSubOffset[i] + numDMInSubs[i]) #particle indices in subhalo
        sub_distances_dm = funcs.dist_vector_nb(sub_pos[i], DM_coords[sub_indices], boxSize) #particle distances to sub center
        inside = np.where(sub_distances_dm <= cut * shmr[i])[0] #particles inside 2shmr
        which_particles[inside + dmInSubOffset[i]] = 1 #mark those particles
        inside_offsets[i] = inside.shape[0] #save number for offsets
    
    inside_offsets = np.cumsum(inside_offsets)
    inside_offsets = np.insert(inside_offsets,0,0)
    return which_particles, inside_offsets

def traceBack_DM(basePath, start_snap, target_snap):
    start = time.time()
    h = il.groupcat.loadHeader(basePath, start_snap)
    boxSize = h['BoxSize'] #ckpc
    h_const = h['HubbleParam']
    
    
    #load all DM particles
    DM_coords = il.snapshot.loadSubset(basePath, start_snap, 1, fields = ['Coordinates']) #ckpc
    
    #load subhalo positions
    sub_pos = il.groupcat.loadSubhalos(basePath, start_snap, fields = ['SubhaloPos']) #ckpc
    
    # load DM particle offsets in subhalos
    g = h5py.File(basePath[:-6] + 'postprocessing/offsets/offsets_' + str(start_snap).zfill(3) + '.hdf5','r')
    
    if not g.__contains__('Subhalo'):
        g.close()
        raise ValueError(f'No Subhalos at snapshot {start_snap}!')
        
    dmInSubOffset = g['Subhalo/SnapByType'][:,1]
    g.close()
    
    numDMInSubs = il.groupcat.loadSubhalos(basePath, start_snap, fields = ['SubhaloLenType'])[:,1]
    shmr = il.groupcat.loadSubhalos(basePath, start_snap, fields = ['SubhaloHalfmassRadType'])[:,4] #ckpc
    
    cut = 2 #only trace back DM particles within two stellar halfmass radii
    
    which_particles, inside_offsets = DM_in_2_shmr(DM_coords, sub_pos, dmInSubOffset, numDMInSubs, shmr, cut, boxSize) #do all calculations in ckpc!
    
    trace_back_indices = np.nonzero(which_particles)[0]
    
    if trace_back_indices.shape[0] == 0:
        raise ValueError('No particles to trace back!')
        
    del DM_coords, sub_pos, shmr, which_particles
    
    ##### now trace those dm particles back: #####
    # only one 'tracer' per dm particle (IDs don't change with time)
    
    dmIDs = il.snapshot.loadSubset(basePath, start_snap, 1, fields = ['ParticleIDs'])
    dmIDs = dmIDs[trace_back_indices] #IDs to find at target snapshot
    
    dmIDs_target_snap = il.snapshot.loadSubset(basePath, target_snap, 1, fields = ['ParticleIDs'])
    
    _, target_DM_inds = match_general(dmIDs, dmIDs_target_snap, is_sorted = False)
    
    f = h5py.File('/vera/ptmp/gc/olwitt/dm/' + basePath[32:39] + f'/dm_indices_{target_snap}.hdf5','w')
    f.create_dataset('dm_indices', data = target_DM_inds)
    f.create_dataset('dmInSubOffset', data = inside_offsets)
    f.close()
    done = time.time()
    print('time to run: ', done - start)
    return #target_DM_inds, inside_offsets
=======
    return insituStarsInSubOffset
>>>>>>> de57ecd0d8a38fa9869bb489726e23ead107bc54
