'''
Created on 23 juil. 2015

@author: admin
'''
import numpy as np
import os.path
import scipy.sparse as sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import Normalizer
from sklearn.externals import joblib
from sklearn.utils.extmath import safe_sparse_dot

# Internal utilities
from _hexagonal import array2hex, hex2array, hex_distance, hex_dqd

# Import the Cythonized module
from csom import c_get_bmu, c_single_unit_deltas, fast_csr_minus_csr

def debug(alg, string):
    print(alg,string)


class SparseSOMMapper(BaseEstimator, ClassifierMixin):
    '''
    This mapper class contains the method for training a self-organized map
    from a given training set. Inspired from the PyMVPA framework SOM Mapper
    class.

    Parameters
    ----------
    kshape : (int, int)
        Shape of the internal Kohonen layer. Currently, only 2D Kohonen
        layers are supported, although the length of an axis might be set
        to 1.
    niter : int
        Number of iteration during network training.
    learning_rate : float
        Initial learning rate, which will continuously decreased during
        network training.
    initialization_func: callable or None
        Initialization function to set self.K_, that should take one
        argument with training samples and return an numpy array. If None,
        then values in the returned array are taken from a standard normal
        distribution.
    topology: string
        Defines the topology of the map in the output space. Currently is the
        square and hexagonal topologies.
    '''

    def __init__(self, kshape=(20,20), n_iter=100, learning_rate=0.05,
                 initialization_func=None, topology="rect"):
        self.kshape = kshape
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self._initialization_func = initialization_func
        self.topology = topology

    def _initialize_fit(self, X, K_init=None, survey=False):
        """ Initialize the data of the estimator.
        Perform some attributes consistency checking and
        initialize the Kohonen map. Allows for the user to provide 
        an existing K_matrix as initialized version. 
        The provided map must match with the kshape and n_features attributes
        """
        # Initialize the dqd matrix for the kernel in the output space
        # scalar for decay of learning rate and radius across all iterations
        self._dqd = np.fromfunction(lambda x,y: (x**2+y**2)**0.5,
                                    self.kshape, dtype='float')
        self.radius = np.max(self.kshape)
        self.iter_scale = self.n_iter / np.log(self.radius)
        X_features = X.shape[1]

        # Initialize the Kohonen Matrix for chaining the training process
        # This can be used for pre-initializing the Kohonen matrix
        # With a sparse PCA component analysis
        K_is_consistent = False
        if K_init != None:
            # Check the consistency of the map with the class shape
            nodes, K_features = K_init.shape
            expected_nodes = self.kshape[0]*self.kshape[1]

            if nodes != expected_nodes:
                K_is_consistent = False
                debug("Kohonen Matrix init",
                      "Unconsistent shapes between the given K_matrix \
                      (%d nodes) and the classifier (%d nodes)"% (nodes,
                        expected_nodes))

            # Check the consistency of the K_matrix with the samples
            if K_features != X_features:
                K_is_consistent = False
                debug("Kohonen Matrix init",
                      "Unconsistent shapes between the given K_matrix \
                      (%d features) and the samples features numbers\
                      (%d features)" % (K_features, X_features))

            # If the shapes are matching the Kohonen matrix is initialized
            if K_is_consistent: self.K_ = K_matrix.tocsr()

        # If no Kohonen map was given or if there was an unconsistency
        # The map is initalized with a sparse random map
        if (K_init is None) or K_is_consistent == False:
            self.K_ = sparse.rand(self.kshape[0]*self.kshape[1],
                                  X_features,
                                  density = 0.1,
                                  format = 'csr')

        # Initialize the fitting algorithm metrics if required
        if survey:
            self.fit_parameters_ = np.zeros(self.n_iter,
                                            dtype = [("it","int"),
                                                    ("radius","f4"),
                                                    ("learning_rate","f4"),
                                                    ("quantization_error","f4"),
                                                    ("unit_deltas_update","f4")])
            self.fit_parameters_["it"] = np.arange(self.n_iter)


    def fit(self, X, y=None, K_init=None, survey=False):
        """Perform network training.

        Parameters
        ----------
        X : CSR Sparse Matrix
            Used for unsupervised training of the SOM.

        K_init : CSR sparse matrix
            Provide a K_matrix for chaining the fit process with a 
            pre-existing map. The shape must match with the requirements
            of the class (n_features from the samples)

        survey: bool
            Turns on the metrics for results analysis. A dataframe is
            created containing different fields describing the evolution
            of the parameters and errors during the training.

        Notes
        -----
        It is assumed that prior to calling this method the _pretrain method
        was called with the same argument.
        """
        # Dimensions of the map
        Xshape = X.shape
        n_features = Xshape[1]
        n_nodes = self.kshape[0]*self.kshape[1]

        # Broadcast the samples over the nodes number (faster training)
        # get rid of this step after cython implementation
        XN = Normalizer().fit_transform(X)
        self._initialize_fit(X, K_init = K_init, survey= survey)

        for it in np.arange(self.n_iter):
            #print it
            # compute the neighborhood impact kernel for this iteration
            k = _compute_influence_kernel(it+1, self._dqd, self.iter_scale,
                                          self.radius, self.learning_rate)
            # extract the matrix
            unit_deltas, quantization_error = batch_unit_deltas(Kohonen = self.K_,
                                                                kernel = k,
                                                                X = X,
                                                                normalized_X = XN,
                                                                kshape = self.kshape,
                                                                Xshape = Xshape)

            # apply cumulative unit deltas
            self.K_ = self.K_ + unit_deltas

            # Metrics analysis of the training process
            if survey:
                self.fit_parameters_["quantization_error"][it]=quantization_error
                #self.fit_parameters_["radius"][it] = self.curr_max_radius
                #self.fit_parameters_["learning_rate"][it] = self.curr_lrate
                self.fit_parameters_["unit_deltas_update"][it]=1.#unit_deltas.dot(unit_deltas.T).sum()


        return self

    def transform(X):
        """ Returns the Kohonen map as representing the output space.
        Parameters:
        -----------
        X: matrix
            CSR matrix of shape = [n_samples, n_features]

        Return:
        -------
        K: csr matrix
            CSR matrix of shape = [n_nodes, n_features] representing
            the fitted data in the INPUT space (features) but aranged
            as in the OUTPUT space (inter node distance)
        """
        if hasattr(self,"K_"):
            return self.K_
        else:
            debug("transform",
                  "The estimator must be fitted before transforming the data" +\
                  "Consider preferably the fit_transform method")
            return None

    def fit_transform(self, X):
        """ Implement the combination of the fit and returning the data.
        Note that the interface is here simplified so that surveying of
        the fitting process is not used.

        Parameters:
        -----------
        X: matrix
            CSR matrix of shape = [n_samples, n_features]

        Return:
        -------
        K: csr matrix
            CSR matrix of shape = [n_nodes, n_features] representing
            the fitted data in the INPUT space (features) but aranged
            as in the OUTPUT space (inter node distance)
        """
        self.fit(X, y = None, survey = False)
        return self.K_

    def predict(self, X):
        """ Predict the position of the samples in the estimator Kohonen a simple
        features check is performed.

        Parameters:
        -----------
        X: matrix
            CSR matrix of shape = [n_samples, n_features]

        Return:
        -------
        Y: array
            Shape =  [n_samples]
            Index of the cluster each sample belongs to.

            """
        if hasattr(self, "K_"):
            KN = Normalizer().fit_transform(self.K_)
            return np.array([c_get_bmu(KN, X.getrow(i))[0] 
                             for i in xrange(X.shape[0])])
        else:
            debug("predict", "The Kohonen network must be trained before use")
            return None

    def fit_predict(self,X):
        """Combination of the fitting and placement of the data from the
        INPUT space shape = (n_features) to the OUTPUT space shape = n_nodes
        """
        self.fit(X, y = None, K_init=None, survey=False)
        return self.predict(X)


def get_bmu(normalized_Kohonen, y):
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
    #similarity = normalized_Kohonen.dot(sampleN.T).toarray()
    similarity = safe_sparse_dot(normalized_Kohonen,sampleN.T)
    loc = np.argmax(similarity)
    return loc, similarity[loc]

def _get_node_coordinates(i_node, ncols, topology = "rect"):
    """Returns the coordinates of the node i_node
    in the output space"""
    i = np.divide(i_node, ncols).astype('int')
    j = i_node % ncols
    return i, j

def _node_to_coordinate(node_index, kshape, topology = "rect"):
    """Return the coordinates of the node in the Kohonen map given the
    node index
    """
    if topology == "rect":
        return np.array((np.divide(node_index, kshape[1]).astype('int'),
                        node_index % kshape[1]))
    else:
        return None

def _get_neighbors(i_node, kshape, topology = "rect", radius = 1):
    """Returns the indices of the array that are to be extracted
    from the index of the K matrix. Used for computation of the
    U-matrix
    """
    nrows, ncols = kshape
    neighbors = []
    if topology == "rect":
        i,j = _node_to_coordinate(i_node, kshape, topology= topology)
        neighbors = [ix*ncols+jx for ix in (i-1,i,i+1)
                             for jx in (j-1,j,j+1)
                             if not(ix==i and jx==j) and
                                 0 <= ix < kshape[0] and
                                 0 <= jx < kshape[1] ]
    return neighbors


def batch_unit_deltas(Kohonen, kernel, X, normalized_X, kshape, Xshape, topology="rect"):
    """Loop over the CSR sample number for computing the batch training on a single
    epoch (training cycle)
    Return:
    -------
    unit_deltas: csr matrix of shape = [n_nodes, n_features]
        Gives for each node the weight update with competing samples (batch)
    quantization_error: float
        Mean of the quantization error for each item
    """
    # Initialize the dimensions
    n_samples, n_features = Xshape
    n_nodes = kshape[0] * kshape[1]

    # Initilaize the sparse matrices
    normalized_Kohonen = Normalizer().fit_transform(Kohonen)
    unit_deltas = sparse.csr_matrix((n_nodes,n_features), dtype=np.double)
    quantization_error_data = np.empty(n_samples, dtype=np.double)

    for i in np.arange(n_samples):
        # Broadcasting the size of the sample over the first axis of K
        sn = normalized_X.getrow(i)

        # Get the location of the matching node for that sample assume rect map
        location, quantization_error_data[i] = c_get_bmu(normalized_Kohonen, sn)
        bmu = _node_to_coordinate(location, kshape, topology=topology)

        # Train all units at once by unfolding the kernel
        infl =_unfold_kernel(bmu, kernel, kshape).flatten()

        # Update the single interation delta as infl[j]*(X[i]-K[j])
        matrix_repr = c_single_unit_deltas(X, Kohonen, i, infl)
        #matrix_repr = sparse.rand(n_nodes,n_features).tocsr()
        unit_deltas = unit_deltas + sparse.csr_matrix(matrix_repr,
                                                      (n_nodes,n_features))

    # Calculate the quantization error as from the distance between samples and
    # their corresponding best matching unit
    quantization_error = quantization_error_data.mean()
    return unit_deltas, quantization_error

def _compute_influence_kernel(iter, dqd, iter_scale,
                              radius,learning_rate):
    """Compute the neighborhood kernel for some iteration.

    Parameters
    ----------
    iter : int
        The iteration for which to compute the kernel.
    dqd : array (nrows x ncolumns)
        This is one quadrant of Euclidean distances between Kohonen unit
        locations.
    iter_scale: float
        Value of the iteration scaling for decaying function of lrate
    radius: integer
        Value of the max radius of the neighborhood function to decay
    learning_rate: float
        Value of the learning rate to decay with epochs

    Return:
    -------
    infl:   sparse csr matrix
    curr_max_radius: current neighborhood kernel radius
    curr_lrate: current value of the learning rate
    """
    # compute radius decay for this iteration
    # same for learning rate
    curr_max_radius = radius * np.exp(-1.0 * iter / iter_scale)
    curr_lrate = learning_rate * np.exp(-1.0 * iter / iter_scale)

    # compute Gaussian influence kernel
    infl = np.exp((-1.0 * dqd) / (2 * curr_max_radius * iter))
    infl *= curr_lrate

    # hard-limit kernel to max radius
    infl[dqd > curr_max_radius] = 0.
    return infl



def _unfold_kernel(bmu, kernel, kshape):
    """Compute the influence matrix for a best matching unit and
    the kernel function. The influence matrix is just a shift of
    the four quadrants of the kernel function, so that infl[b] = 1
    Parameters:
    ----------
    bmu: tuple (x,y)
        Coordinates of the best matching unit in the output space
    kernel: numpy array of shape = kshape
        The kernel has the size of the maximal size of the
        neighborhood influence matrix: the size of the map
    kshape: t-uple, shape of the Kohonen map

    Return:
    -------
    The influence matrix of the kernel, flattened
    """
    infl = np.vstack((
            np.hstack((
                # upper left
                kernel[bmu[0]:0:-1, bmu[1]:0:-1],
                # upper right
                kernel[bmu[0]:0:-1, :kshape[1] - bmu[1]])),
            np.hstack((
                # lower left
                kernel[:kshape[0] - bmu[0], bmu[1]:0:-1],
                # lower right
                kernel[:kshape[0] - bmu[0], :kshape[1] - bmu[1]]))
                    ))
    return infl

def _euclidian_metrics(x,y):
    return x**2 + y**2

def _unfold_kernel_periodic(b, k):
    """Compute the influence matrix for a best matching unit and
    the kernel function. The influence matrix is just a shift of
    the four quadrants of the kernel function, so that infl[b] = 1
    Parameters:
    ----------
    b: tuple
        Coordinates of the best matching unit in the output space
    k: numpy array of shape = kshape
        The kernel has the size of the maximal size of the
        neighborhood influence matrix: the size of the map
    """
    low = np.roll(np.hstack((np.fliplr(k[:,1:]),k)),
                        shift = b[1],
                        axis = 1)
    infl = np.roll(np.vstack((np.flipud(low[1:,:]),low)),
                    shift = b[0],
                    axis = 0)
