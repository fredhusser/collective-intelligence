"""
encoding = utf-8
filename = postprocessing.py
author = Frederic Husser
Description:
This module contains the post-processing tools for working
on the data from the main data analysis pipeline. The raw data 
can be prepared for visualization and for publication into 
a DataBase.
"""

import numpy as np
from scipy import sparse
from sklearn.preprocessing import Normalizer

# Utilities for the processing of 2D maps
from som import debug
from som import _get_node_coordinates, _get_neighbors
from csom import c_get_bmu


def build_H_matrix(X, kohonen):
    """Build the reprentation of the hits matrix from the INPUT space into
    the OUTPUT space, given the sample matrix X.
    Required is to have the same sample set as used for training, that is 
    to say the same n_features dimensions.

    Parameters
    ----------
    X: array-like CSR matrix
        Sparse representation of the samples in the INPUT space in a CSR
        matrix of shape = [n_samples, n_features]. Must be the same as 
        that used for fitting the map.

    Return
    ------
    H_matrix : ndarray
        Numpy array of shape = [n_nodes] of integers giving for each node
        the number of best matching documents
    """
    # Initialize the hits matrix as an ndarray of ints
    debug("Build_H_matrix","Starting the counting of hits...")
    debug("Build_H_matrix","Using %d documents with %d features" % X.shape)

    n_nodes = kohonen.shape[0]
    H_matrix = np.zeros(n_nodes, dtype = np.int)
    KN = Normalizer().fit_transform(kohonen)

    # Get the best matching units for all vectors
    n_samples, n_features = X.shape
    for i in xrange(n_samples):
        bmu_idx = c_get_bmu(KN,X.getrow(i))[0]
        H_matrix[bmu_idx]+=1
        print bmu_idx
    return H_matrix

def build_P_matrix(kohonen, features):
    """Build the projection matrix given a set of features from
    the INPUT space of the Kohonen map. The projection is based 
    on the calculation of the mean of the selected features for
    each node.

    Parameters
    ---------
    features: ndarray of integers
        Numpy array of the features to be selected for the projection
        If None all features are selected.
        
    Return
    ------
    P_matrix: ndarray
        Numpy array of shape n_nodes, dtype= np.double giving the value
        of the mean of the projected samples applied on the normalized 
        Kohonen matrix
    """
    # Normalization of the Kohonen matrix is necessary for validity
    debug("Build_P_matrix", "Starting the projection...")
    KN = Normalizer().fit_transform(kohonen).tolil()
    n_nodes, n_features = KN.shape
    P_matrix = np.zeros(n_nodes)
        
    # Slice over the rows of the matrix and build the projection
    for i in np.arange(n_nodes):
        selected_features = np.intersect1d(features,KN.rows[i])
        data = np.asarray(KN.data[i], dtype=np.double)
        P_matrix[i] = np.mean(data[selected_features])
    return P_matrix
