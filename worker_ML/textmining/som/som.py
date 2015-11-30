import numpy as np
import os.path
import scipy.sparse as sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import Normalizer
from sklearn.utils.extmath import safe_sparse_dot, fast_dot

from csom import parallel_single_unit_deltas, get_distance_metrics

# Internal utilities
#from _hexagonal import array2hex, hex2array, hex_distance, hex_dqd

def debug(alg, string):
    print(alg,string)
    
class SOMMapper(BaseEstimator, ClassifierMixin):
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
    def __init__(self, kshape=(20,20), n_iter=100, learning_rate=0.1,
                 initialization_func=None, infl_decay_type="default", 
                 topology="rect"):
        self.kshape = kshape
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self._initialization_func = initialization_func
        self.topology = topology
        self.infl_decay_type = infl_decay_type
        
    def _initialize_fit(self, X, K_init=None):
        """ Initialize the data of the estimator.
        Perform some attributes consistency checking and
        initialize the Kohonen map. Allows for the user to provide 
        an existing K_matrix as initialized version. 
        The provided map must match with the kshape and n_features attributes
        """
        self._dqd = np.fromfunction(lambda x,y: (x**2+y**2)**0.5,
                                    self.kshape, dtype='float')
        self.radius = np.min(self.kshape)
        self.iter_scale = self.n_iter / np.log(self.radius)
        self.n_nodes = self.kshape[0]*self.kshape[1]
        self.quantization_error = np.empty(self.n_iter)
        
        self.learning_rate_f = 0.05*self.learning_rate
        self.sigma = np.sqrt(self.n_nodes)
        self.sigma_f = 0.2
        
        # Check the initialized Kohonen map shape consistency
        check_K_shape = False
        if K_init is not None:
            check_K_shape = ((K_init.shape[0] == self.n_nodes) and
                            (K_init.shape[1] == X.shape[1]))
        
        # Assign the initialized map 
        if check_K_shape:
            self.K_ = np.array(K_init, dtype= np.double)
        else:
            self.K_ = np.random.rand(self.n_nodes, X.shape[1])
            self._clean_kohonen()
        self.KN_ = Normalizer().fit_transform(self.K_)
            
    def fit(self, X, y=None, K_init=None):
        """Perform network training.

        Parameters
        ----------
        X : CSR Sparse Matrix
            Used for unsupervised training of the SOM.

        K_init : CSR sparse matrix / ndarray
            Provide a K_matrix for chaining the fit process with a
            pre-existing map. The shape must match with the requirements
            of the class (n_features from the samples)

        sparse: bool
            If the sample matrix is to processed as a sparse matrix

        Notes
        -----
        It is assumed that prior to calling this method the _pretrain method
        was called with the same argument.
        """
        if sparse.issparse(X):
            X = X.toarray()
            
        self._initialize_fit(X, K_init=K_init)
        n_samples, n_features = X.shape
        XN = Normalizer().fit_transform(X)
        topological_errors = []
        node_to_coordinates_table = np.array([[np.divide(node_index, self.kshape[1]).astype('int'),
                                              node_index % self.kshape[1]]
                                              for node_index in xrange(self.n_nodes)])
        for it in np.arange(self.n_iter):
            #print it
            unit_deltas = np.zeros((self.n_nodes,n_features), dtype=np.double)
            quantization = np.empty(n_samples, dtype=np.double)
            kernel = self._compute_influence_kernel(it+1)
            for s in xrange(n_samples):    
                node_index, quantization[s]= self.get_bmu(XN[s,:])
                
                x,y = node_to_coordinates_table[node_index,:]
                
                infl =self._fast_unfold_kernel(x,y, kernel).flatten()
                
                unit_deltas+=parallel_single_unit_deltas(X[s,:], self.K_, infl, self.n_nodes, 1e-6)
                #unit_deltas+= (X[i,:] - Kohonen)*infl[:,np.newaxis]
                
            self.quantization_error[it] = quantization.mean(axis=0)            
            self.K_ += unit_deltas
            self.KN_ = Normalizer().fit_transform(self.K_)
        
        self._clean_kohonen()
        self.measure_topological_error(X)
        return self

        
    def transform(self):
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

    def fit_transform(self, X, **kwargs):
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
        self.fit(X, **kwargs)
        return self.transform()

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
            Index of the node each sample belongs to.

            """
        if X.ndim <2: X=X[np.newaxis,:]
        XN = Normalizer().fit_transform(X)
        if hasattr(self, "KN_"):
            return np.array([self.get_bmu(XN[i,:])[0] 
                             for i in xrange(X.shape[0])])
        else:
            debug("predict", "The Kohonen network must be trained before use")
            return None

    def fit_predict(self,X, **kwargs):
        """Combination of the fitting and placement of the data from the
        INPUT space shape = (n_features) to the OUTPUT space shape = n_nodes
        """
        self.fit(X, **kwargs)
        return self.predict(X)
    
    def _node_to_coordinate(self, node_index):
        """Return the coordinates of the node in the Kohonen map given the
        node index
        """
        if self.topology == "rect":
            return np.array((np.divide(node_index, self.kshape[1]).astype('int'),
                            node_index % self.kshape[1]))
        else:
            return None
    
    def _clean_kohonen(self, epsilon = 1e-4):
        pass
        #self.K_[self.K_<epsilon] = 0.
        
    def measure_topological_error(self, samples):
        """Known as the average distance between two best matching units
        """
        n_samples = samples.shape[0]
        topological_error = np.empty(n_samples, dtype=np.double)
        for i in np.arange(n_samples):
            bmu1,bmu2 = get_second_bmu(self.K_, samples[i,:])
            x1,y1 = self._node_to_coordinate(bmu1)
            x2,y2 = self._node_to_coordinate(bmu2)
            topological_error[i]=np.sqrt((x1-x2)**2+(y1-y2)**2)
        self.topological_error = topological_error.mean()

    def get_bmu(self, yn):
        """Returns the ID of the best matching unit.
        Best is determined from the cosine similarity of the
        sample with the normalized Kohonen network.
    
        See https://en.wikipedia.org/wiki/Cosine_similarity
        for cosine similarity documentation.
        TODO: make possible the finding the second best matching unit
    
        Parameters
        ----------
        KN : sparse matrix
            Shape = [n_nodes, n_features] must be normalized according to
            l2 norm as used in the sklearn Normalizer()
        y : vector of dimension 1 x nfeatures
            Target sample.
    
        Returns
        -------
        tuple : (loc, cosine_distance)
            index of the matching unit, with the corresponding cosine distance
        """
        #d = ((self.K_-y)**2).sum(axis=1)
        #loc = np.argmin(d)
        #qe = np.sqrt(d[loc])
        similarity = fast_dot(self.KN_, yn.T)
        loc = np.argmax(similarity)
        qe = 1/(1.0e-4+similarity[loc])-1
        return loc, qe

    def _compute_influence_kernel(self, iter):
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
        curr_max_radius = self.radius * np.exp(-1.0 * iter / self.iter_scale)
        #curr_max_radius = self.radius * iter/self.n_iter
        if self.infl_decay_type == "default":
            curr_lrate = self.learning_rate * np.exp(-1.0 * iter / self.iter_scale)
            infl = np.exp((-1.0 * self._dqd**2) / (2 * curr_max_radius**2))
            infl *= curr_lrate
            
        if self.infl_decay_type == "sigma":
            curr_lrate = ((self.learning_rate_f/self.learning_rate)**(iter/self.n_iter))
            curr_lrate *= self.learning_rate
            sigma = self.sigma*((self.sigma_f/self.sigma)**(iter/self.n_iter))
            infl = np.exp((-1.0 * self._dqd**2) / sigma**2)
            infl *= curr_lrate
    
        # hard-limit kernel to max radius
        infl[self._dqd > curr_max_radius] = 0.
        return infl
    
    def _fast_unfold_kernel(self,x,y,kernel):
        kx,ky=self.kshape
        infl = np.empty((kx,ky), dtype=np.double)
        infl[:x, :y] = kernel[x:0:-1, y:0:-1]
        infl[:x, y:] = kernel[x:0:-1, :ky - y]
        infl[x:, :y] = kernel[:kx - x, y:0:-1]
        infl[x:, y:] = kernel[:kx - x, :ky - y]
        return infl

def get_second_bmu(KN, y):
    """Function used for measuring the regularity of the mapping from 
    the INPUT into the OUTPUT space, defined as the distance between the
    two best matching units for a sample. The mean of this measure
    for all the units is the topological error. 
    """
    d = np.sum((KN-y)**2, axis=1)
    bmus = np.argsort(d)
    return bmus[0], bmus[1]
    

    
def build_U_matrix(kohonenMatrix, kshape, topology):
    """Calculates the Unified Distance matrix given the Kohonen
    map in the input space: expressed as a list of nodes and features.
    The distances are infered from the distance matrix in the
    output space. Only the nearest neighbors are selected.

    Parameters:
    -----------
    kohonenMatrix: ndarray
        The matrix upon which to build the U matrix
        Shape = [n_nodes, n_features]
    Return:
    -------
    U_matrix : ndarray
        Matrix giving for each node the mean distance to its
        neighbors: Shape = [n_nodes]
    """

    # The matrix must be a lil matrix for fast iteration over rows
    debug("Build_U_matrix", "Starting the calcuation of U_matrix...")
    KN = Normalizer().fit_transform(kohonenMatrix)

    #Initilaize the U_matrix as a numpy array
    n_nodes, n_features = kohonenMatrix.shape
    U_matrix = np.zeros(n_nodes)

    # Iterate over the rows (nodes): The Value of the U_Matrix 
    # at the node coordinate is the mean of the distances with all neighbors
    for i_node in np.arange(n_nodes):
        neighbors = np.sort(_get_neighbors(i_node, kshape, topology=topology))
        U_matrix[i_node] = (KN[i_node,:]*KN[neighbors,:]).sum(axis=1).mean()
    return U_matrix

def _get_node_coordinates(i_node, ncols, topology = "rect"):
    """Returns the coordinates of the node i_node
    in the output space"""
    i = np.divide(i_node, ncols).astype('int')
    j = i_node % ncols
    return i, j

def _get_neighbors(i_node, kshape, topology = "rect", radius = 1):
    """Returns the indices of the array that are to be extracted
    from the index of the K matrix. Used for computation of the
    U-matrix
    """
    nrows, ncols = kshape
    neighbors = []
    if topology == "rect":
        i = np.divide(i_node, ncols).astype('int')
        j = i_node % ncols
        neighbors = [ix*ncols+jx for ix in (i-1,i,i+1)
                             for jx in (j-1,j,j+1)
                             if not(ix==i and jx==j) and
                                 0 <= ix < kshape[0] and
                                 0 <= jx < kshape[1] ]
    return neighbors




if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from sklearn.feature_extraction.image import grid_to_graph
    from sklearn.cluster import AgglomerativeClustering
    kshape = (30,20)
    n_iter = 100
    learning_rate = 0.01
    n_colors = 100
    
    spcolors = np.random.rand(n_colors,3)
    mapper = SOMMapper(kshape=kshape, n_iter=n_iter, learning_rate=learning_rate)
    kohonen = mapper.fit_transform(spcolors)
    U_Matrix = build_U_matrix(kohonen, kshape, topology="rect")

    
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.imshow(np.split(kohonen, kshape[0], axis=0))
    ax1.set_title("Kohonen Map")
    
    ## Clustering
    n_clusters = 5  # number of regions
    connectivity = grid_to_graph(kshape[0],kshape[1])
    ward = AgglomerativeClustering(n_clusters=n_clusters,
            linkage='ward', connectivity=connectivity).fit(kohonen)
            
    label = np.reshape(ward.labels_, kshape)
    for l in range(n_clusters):
        ax1.contour(label == l, contours=1,
                    colors=[plt.cm.spectral(l / float(n_clusters)), ])
    
    ax2 = fig.add_subplot(221)
    ax2.imshow(np.split(U_Matrix, kshape[0], axis=0))
    ax2.set_title("U_Matrix")
    plt.show()