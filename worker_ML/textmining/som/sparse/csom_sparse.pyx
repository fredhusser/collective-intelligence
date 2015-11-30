# cython: profile=True
# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE=1

# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
import numpy as np
import scipy.sparse as sp   
cimport numpy as np
cimport cython

from scipy import sparse
from scipy.sparse._sparsetools import csr_minus_csr, csr_plus_csr
from sklearn.preprocessing import Normalizer


np.import_array()
DTYPE = np.int
ctypedef np.int_t DTYPE_t
ctypedef np.float64_t DOUBLE
ctypedef np.int32_t INT


#cdef extern from "./src/cblas/cblas.h":
#    double ddot "cblas_ddot"(int N, double *X, int incX, double *Y, int incY)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef c_get_bmu(normalized_Kohonen, y):
    """Returns the ID of the best matching unit.
    Best is determined from the cosine similarity of the
    sample with the normalized Kohonen network.

    See https://en.wikipedia.org/wiki/Cosine_similarity
    for cosine similarity documentation.
    TODO: make possible the finding the second best matching unit

    Parameters
    ----------
    normalized_Kohonen : sparse matrix
        Shape = [n_nodes, n_features] must be normalized according to
        l2 norm as used in the sklearn Normalizer()
    y : vector of dimension 1 x nfeatures
        Target sample.

    Returns
    -------
    tuple : (loc, cosine_distance)
        index of the matching unit, with the corresponding cosine distance
    """
    # The dot product of the vector with each node is computed
    sampleN=Normalizer().fit_transform(y)
    similarity = normalized_Kohonen.dot(sampleN.T).toarray()
    loc = np.argmax(similarity)
    return loc, similarity[loc]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(False)
cpdef c_single_unit_deltas(X, K, int i, np.ndarray[DOUBLE, ndim=1] influence):
    """ Apply the weight update on a single sample i of matrix
    X with respect to the Kohonen matrix K.

    Returns:
    --------
    All parameters for building a csr matrix (data, indices, indptr, shape)
    as being the result of infl[j]*(X[i]-K[j]) for j a node
    """
    cdef:
        # Initialize the single unit deltas matrix as its sparse representation
        np.ndarray[DOUBLE, ndim = 1] X_data
        np.ndarray[DTYPE_t, ndim = 1] X_indices, X_indptr

        # Initialize the Unit_deltas output
        np.ndarray[DOUBLE, ndim=1] deltas_data = np.array([], dtype=np.double)
        np.ndarray[DTYPE_t, ndim=1] deltas_indices = np.array([], dtype=DTYPE)
        np.ndarray[DTYPE_t, ndim=1] deltas_indptr = np.array([0], dtype=DTYPE)

        # Define the type of the slice of the K matrix used for vectors
        np.ndarray[DOUBLE, ndim = 1] K_data_node
        np.ndarray[DTYPE_t, ndim = 1] K_indices_node, K_indptr_node
        np.ndarray[DOUBLE, ndim = 1] sum_data
        np.ndarray[DTYPE_t, ndim = 1] sum_indices, sum_indptr

        # Define the indices for loops
        INT new_indptr, p0, p1, n0, n1, n_nodes, n_features, max_nnz

    # Get the slice of the Sample item: get X[i]
    n_nodes, n_features = K.shape
    p0 = X.indptr[i]
    p1 = X.indptr[i+1]
    X_data = X.data[p0:p1]
    X_indices = X.indices[p0:p1]
    X_indptr = np.array([0,p1-p0])
    new_indptr = 0
    
    for j in range(n_nodes):
        
        if influence[j] > 0.001:
            # Get the slice of the Kohonen matrix for a given sample
            n0 = K.indptr[j]
            n1 = K.indptr[j+1]
    
            # Internal structure of the slice of the Kohonen matrix
            K_data_node = K.data[n0:n1]
            K_indices_node = K.indices[n0:n1]
            K_indptr_node = np.array([0, n1-n0])
            max_nnz = p1-p0 + n1-n0 # = to len(X_data) + len(K_data_node)
    
            # Perform the substraction of two 1D sparse vectors
            # Make the X[i] - K[j]
            sum_data, sum_indices, sum_indptr  = \
                        fast_csr_minus_csr(1, n_features, max_nnz,
                                           X_data,
                                           X_indices,
                                           X_indptr,
                                           K_data_node,
                                           K_indices_node,
                                           K_indptr_node)
    
            # Update the csr deltas matrix with the new node
            new_indptr = new_indptr + sum_indptr[1]
    
            # Multiply the data with the influence matrix
            sum_data = sum_data*influence[j]
            
            # Update the CSR values
            deltas_data = np.concatenate((deltas_data, 
                                          sum_data[:sum_indptr[1]]))
            deltas_indices = np.concatenate((deltas_indices, 
                                             sum_indices[:sum_indptr[1]]))

        # Return the comprehensive version of the deltas sparse matrix
        deltas_indptr = np.concatenate((deltas_indptr, 
                                        np.array([new_indptr], dtype=DTYPE)))


    return deltas_data, deltas_indices, deltas_indptr

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef fast_csr_minus_csr(INT n_nodes, INT n_features, INT max_nnz,
                                np.ndarray[DOUBLE, ndim=1] X_data,
                                np.ndarray[DTYPE_t, ndim=1] X_indices,
                                np.ndarray[DTYPE_t, ndim=1] X_indptr,
                                np.ndarray[DOUBLE, ndim=1] Y_data,
                                np.ndarray[DTYPE_t, ndim=1] Y_indices,
                                np.ndarray[DTYPE_t, ndim=1] Y_indptr):
    """ Attempt to make a cython wrapper of the substraction operated within
    scipy sparse tools. Apply the binary operation fn to two sparse matrices.
    """
    # Define the types of the output
    cdef:
        np.ndarray[DOUBLE, ndim=1] data
        np.ndarray[DTYPE_t, ndim=1] indices, indptr
        
    data = np.empty(max_nnz, dtype=np.double)
    indptr = np.empty(len(X_indptr), dtype=DTYPE)
    indices = np.empty(max_nnz, dtype=DTYPE)

    csr_minus_csr(n_nodes, n_features,
                   X_indptr, X_indices, X_data,
                   Y_indptr, Y_indices, Y_data,
                   indptr, indices, data)
    return data, indices, indptr