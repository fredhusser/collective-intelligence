'''
Created on 24 juil. 2015

@author: admin
'''
import unittest

import _hexagonal as hex
import numpy as np

import som


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
        self.my_som = som.SOMMapper(self.kshape, self.niter)

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
            unit, metrics = som.get_bmu(KN, K[test, :])
            self.assertEqual(unit,test)


    def testPretrain(self):
        """Test the consistency of the initialized 
        Kohonen map.
        """
        X = np.array([[0,1,2],
                       [3,4,5],
                       [6,7,8],
                       [9,10,11]],dtype = 'float')
        mysom = som.SOMMapper(n_iter=0, kshape=(10, 20))
        mysom.fit(X)        
        self.assertEqual(mysom.K_.shape,(200,3), "Shape mismatch")
    
    
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
        K = np.array([[1,0,0],[1,1,0],[1,1,0]]).flatten()
        
        #Create a sample SOM Mapper
        mysom = som.SOMMapper(kshape=(3, 3))
        U_ist = som.build_U_matrix(K[:, np.newaxis], mysom.kshape, mysom.topology)
        U_soll = np.array([[2./3.,0,0], [4./5.,0.5,0], [1,3./5.,0]]).astype(float).flatten()
        self.assert_((U_ist == U_soll).all(), )#U_ist.tostring() +'\n' + U_soll.tostring())

    def testUnfoldKernel(self):
        # Test the unfolding of a kernel so that the shape match 
        bmu = [3,3]
        kshape = (10,10)
        
        # Test with a simple kernel being the sum of the coordinates
        kernel = np.fromfunction(lambda x,y: x+y, kshape)
        infl = som._fast_unfold_kernel(bmu[0], bmu[1], kernel, kshape)
        infl = infl.flatten()
        self.assertEqual(infl.size, kshape[0]*kshape[1])
        self.assertEqual(infl[4*10+4],2.)
        self.assertEqual(infl[5*10+5],4.)
        self.assertEqual(infl[bmu[0]*10+bmu[1]], 0.)


    def testSingleDeltas(self):
        X = np.array([[0,0,0],
                    [1,1,1],
                    [2,2,2],
                    [4,4,4]]).astype(np.double)
        K = np.array([[0,0,0],
                    [1,1,1],
                    [2,2,2]]).astype(np.double)
        infl = np.array([1,1,1,1]).astype(np.double)
        n_nodes = 3
        for i in range(4):
            result = np.array(som.parallel_single_unit_deltas(X[i, :], K, infl, n_nodes, 0.1))
            control = X[i,:]-K
            #print "\nX[%d] - K\n" % i
            #print "Result\n", np.array(result)
            #print "Control\n", control
            self.assert_((result==control).all())
            

        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()