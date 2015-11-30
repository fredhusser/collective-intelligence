"""Helper module for hexagonal frames.
Compute the distance matrices and manage the conversion from
and to the hexagonal frames.
"""
import numpy as np


def floor2(x):
    return np.sign(x)*np.floor(np.abs(x))

def ceil2(x):
    return np.sign(x)*np.ceil(np.abs(x))

def array2hex(coord):
    """Computes the coordinates change of a point in the 
    hexagonal array, to the hexagonal topology.
    Parameters:
    -----------
    A: ndarray of shape (i,j)
        Indices of the array of hexagons
    
        Returns:
    -------
    X: ndarray
        Indices of the elements in the hexagonal frame
    """
    coord = np.array(coord, ndmin = 2)
    hx = np.zeros_like(coord)
    hx[...,0] = coord[...,0] - floor2(coord[...,1]/2.).astype(int)
    hx[...,1] = coord[...,0] + ceil2(coord[...,1]/2.).astype(int)
    return hx

def hex2array(coord):
    """Reverse function giving the coordinates expressed
    in the array of hexagons of a point in the hexagonal
    frame.
    Parameters:
    -----------
    (xh,yh): tuple of int
        Indices of the array of hexagons
    
    Returns:
    -------
    (i,j): tuple of int
        Indices of the elements in the array of hexagons
    """
    coord = np.array(coord, ndmin = 2)
    ar = np.zeros_like(coord)
    ar[...,0] = np.array(floor2(coord.sum(axis=coord.ndim-1)/2.).astype(int))
    ar[...,1] = np.array(coord[...,1]-coord[...,0])
    return ar

def hex_distance(A,B):
    """Returns the distance between A and B derived from their
    coordinates in the hexagonal frame.
    Parameters:
    -----------
    A: tuple (xa,ya)
        coordinates of the point A
    B: tuple (xb,yb)
        coordinates of B
    """
    A=np.array(A, ndmin=2)
    B=np.array(B, ndmin=2)
    D=B-A
    daxis = D.ndim-1
    return np.where(D.prod(axis=daxis)>=0,
                    np.max(np.abs(D), axis=daxis),
                    np.abs(D).sum(axis= daxis))


def hex_dqd(shape, origin=(0,0)):
    """Creates an array of distances expressed in the hexagonal
    frame for an array of hexagons.
    Parameters:
    -----------
    shape: tuple (x,y)
        size of the array of hexagons
    
    origin: tuple(x0,y0)
        used for computing the distances from the center of the grid
        Set to (0,0) if only one quadrant is precomputed.

    Returns:
    --------
    dqd: ndarray of shape (x,y)
        Numpy array of distances to the origin.
    """
    # Express the coordinates in the hex frame
    xy_mesh = np.dstack(np.fromfunction(lambda i,j: [i,j], shape))
    dqd = hex_distance(array2hex(np.array(origin)), array2hex(xy_mesh))
    return dqd
