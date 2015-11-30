#### cython: profile=False
#### cython: linetrace=False
#### distutils: define_macros=CYTHON_TRACE=1

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

np.import_array()
DTYPE = np.int
ctypedef np.int_t DTYPE_t
ctypedef np.float64_t DOUBLE
ctypedef np.int32_t INT

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(False)
cpdef parallel_single_unit_deltas(double[:] Xi, double[:,:] K, double[:] infl, 
                                  Py_ssize_t n_nodes, double infl_epsilon=0.1):
    cdef:
        double[:,:] unit_delta = np.zeros((n_nodes, Xi.shape[0]), dtype=np.double)
        Py_ssize_t j, k, n_features=Xi.shape[0]
        
    for j in prange(n_nodes, nogil=True):
        if infl[j]>infl_epsilon:
            for k in prange(n_features):
                unit_delta[j,k] = (Xi[k] - K[j,k])*infl[j]
    return unit_delta

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(False)
cpdef get_distance_metrics(double[:,:] KN, double[:] y,
                           Py_ssize_t n_nodes, Py_ssize_t n_features):
    cdef:
        double[:] distances = np.zeros(n_nodes, dtype=np.double)
        Py_ssize_t j, k
                
    for j in prange(n_nodes, nogil=True):
        distances[j] = 0.
        for k in prange(n_features):
            distances[j] += (KN[j,k]-y[k])**2
    return distances