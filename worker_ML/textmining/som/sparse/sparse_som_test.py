'''
Created on 24 juil. 2015

@author: admin
'''
import unittest
import numpy as np
import scipy.sparse as sparse
import som
import csom
import _hexagonal as hex

class SOMMaperTest(unittest.TestCase):
    def setUp(self):
        """
        We create a simple test corpus that can be used as text data.
        """
        text1 = 'The simple sparse self-organizing map mapper'
        text2 = 'The algorithm uses the sparse'
        text3 = 'The cat is playing.'
        self.test_array = np.array([text1,text2,text3]) 
        self.kshape = (10,10)
        self.niter = 10
        self.my_som = som.SOMMapper(self.kshape,self.niter)

    def tearDown(self):
        pass

    def testGetBMU(self):
        """Tests the looking for the best matching unit with a 
        random map by extracting the vectors from the map as 
        samples which should match with themselves.
        """
        K = np.random.rand(100,1000)
        KN = som.Normalizer().fit_transform(K)

        for test in np.random.randint(0,100, size=100):
            unit, metrics = som.get_bmu(KN, K[test,:])
            self.assertEqual(unit,test)


    def testPretrain(self):
        """Test the consistency of the initialized 
        Kohonen map.
        """
        X = np.array([[0,1,2],
                       [3,4,5],
                       [6,7,8],
                       [9,10,11]],dtype = 'float')
        mysom = som.SOMMapper(n_iter=0, kshape=(10,20))
        mysom.fit(X)        
        self.assertEqual(mysom.K_.shape,(200,3), "Shape mismatch")
    
    def testSamplesBroadcasting(self,nsamples=300, nfeatures=1000, nnodes= 100):
        """Broadcast a random sample set over a number of features. Check the
        shape of the output as well as the regularity of the samples.
        """
        samples = sparse.rand(nsamples, nfeatures, density=0.01, format="csr")
        broad = som._sample_broadcast(samples, nnodes)
        
        # The shape of the broadcast samples set must meet requirements
        self.assertEqual(len(broad),nsamples)
        self.assertEqual(broad[0].shape,(nnodes,nfeatures))

        # Check the regularity of the samples of first two items in the output
        self.assertTrue((broad[0].getrow(0).data==broad[0].getrow(1).data).all())

        # Check that the output list matches with the inputs
        self.assertTrue((broad[0].getrow(0).data == samples.getrow(0).data).all())
        self.assertTrue((broad[nsamples-1].getrow(0).data == samples.getrow(nsamples-1).data).all())


    
    def testHexDistance(self):
        """Test the random mapping function from a grid of hexagons to
        and back the hexagonal frame. The hexagonal frame is used for 
        precomputing the neighborhood function for an array of hexagons
        """
        mesh = np.dstack(np.fromfunction(lambda i,j: [i,j], (3,3)))

        print hex.hex_dqd((4,4))

        array_test = [[0,2],
                      [0,1],
                      [1,0],
                      [0,3]]
        hex_test = np.array([[-1,1],
                    [0,1],
                    [1,1],
                    [-1,2]])                      
        # The transformations should be invertible
        mesh_hex = hex.array2hex(mesh)
        mesh_back = hex.hex2array(mesh_hex)
        self.assertTrue((mesh==mesh_back).all())

        # Test some random values in the 
        self.assertTrue((hex.array2hex(array_test)==hex_test).all())
        
        # The shapes in the hex and array frame must match
        self.assertEqual(mesh_hex.shape, mesh.shape)

        # Test the distance for a given vector in hex frame
        A = hex.hex_distance([3,4],[[0,0],[5,1]])
        self.assertTrue((A==np.array([4,5])).all())

    def testUmatrix(self):
        K = np.array([[1,0,0],[1,1,0],[1,1,0]])
        Ks = sparse.csr_matrix(K.flatten()[:,np.newaxis])
        
        #Create a sample SOM Mapper
        mysom = som.SOMMapper(kshape=(3,3))
        U_ist = mysom.get_U_Matrix(Ks, is_normalized = True)
        U_soll = np.array([[2./3.,0,0], [4./5.,0.5,0], [1,3./5.,0]]).astype(float)
       
        self.assert_((U_ist == U_soll).all(), )#U_ist.tostring() +'\n' + U_soll.tostring())

    def testUnfoldKernel(self):
        # Test the unfolding of a kernel so that the shape match 
        bmu = [3,3]
        kshape = (10,10)
        
        # Test with a simple kernel being the sum of the coordinates
        kernel = np.fromfunction(lambda x,y: x+y, kshape)
        infl = som._unfold_kernel(bmu = bmu, kernel = kernel, kshape = kshape)

        self.assertEqual(infl.shape, kshape)
        self.assertEqual(infl[4,4],2.)
        self.assertEqual(infl[5,5],4.)
        self.assertEqual(infl[bmu[0], bmu[1]], 0.)

    def testSparseVectorSubstraction(self):
        # Define the test vector
        X = sparse.csr_matrix(np.arange(10)*2)
        Y = sparse.csr_matrix(np.arange(10))

        # Perform Z = X - Y = 2*Y - Y = Y
        data, ind,indptr, shape = som._1D_sparse_vector_sub(X.data, X.indices, Y.data, Y.indices,X.shape)
        Z = sparse.csr_matrix((data,ind,indptr))
        self.assert_((Y.data==data).all())
        self.assert_((Y.indices==ind).all())
        self.assert_((Y.indptr==indptr).all())

        # Test with arbitrary vector
        X1 = sparse.csr_matrix(np.array([0,2,0,6,8,0]))
        X2 = sparse.csr_matrix(np.array([1,0,4,7,10,0]))
        X3 = sparse.csr_matrix(np.array([-1,2,-4,-1,-2,0]))
        data, ind,indptr, shape = som._1D_sparse_vector_sub(X1.data, X1.indices, X2.data, X2.indices, X1.shape)
        Z = sparse.csr_matrix((data,ind,indptr), shape=shape)
        self.assert_((X3.data==data).all())
        self.assert_((X3.indices==ind).all())
        self.assert_((X3.indptr==indptr).all())


    def testSingleDeltas(self):
        X = sparse.csr_matrix(np.array([[0,0,0],
                                        [1,1,1],
                                        [2,2,2],
                                        [4,4,4]]))
        K = sparse.csr_matrix(np.array([[0,0,0],
                                        [1,1,1],
                                        [2,2,2]]))
        for i in range(4):
            data, ind, indptr, shape = som.single_unit_deltas(i, X, K)
            result = sparse.csr_matrix((data,ind,indptr),shape)
            control = sparse.vstack((X[i],X[i],X[i]))-K
            print "\nX[%d] - K\n" % i
            print "Result\n", result.todense()
            print "Control\n", control.todense()

    def test_csr_minus_csr(self):
        # Define the test matrices
        A = sparse.csr_matrix(np.arange(10), dtype=np.double)
        B = sparse.csr_matrix(np.arange(10)*2, dtype = np.double)
        BmA = csom.fast_csr_minus_csr(A.shape[0],A.shape[1],A.nnz+B.nnz,
                                     B.data,B.indices,B.indptr,
                                     A.data,A.indices,A.indptr)
        print BmA
        print A.indices
        C = sparse.csr_matrix(BmA,A.shape)
        print "C", C.todense()

        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()