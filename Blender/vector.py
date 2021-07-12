#!/usr/bin/env python
#*****************
#Vector Operations
#*****************
#
#*Author:* Dustin Kleckner (2014)

import numpy as np

#------------------------------------------------------------------------------
# Basic Operations
#------------------------------------------------------------------------------
def mag(X):
    '''Calculate the length of an array of vectors.'''
    return np.sqrt((np.asarray(X)**2).sum(-1))

def mag1(X):
    '''Calculate the length of an array of vectors, keeping the last dimension
    index.'''
    return np.sqrt((np.asarray(X)**2).sum(-1))[..., np.newaxis]

def dot(X, Y):
    '''Calculate the dot product of two arrays of vectors.'''
    return (np.asarray(X)*Y).sum(-1)

def dot1(X, Y):
    '''Calculate the dot product of two arrays of vectors, keeping the last
    dimension index'''
    return (np.asarray(X)*Y).sum(-1)[..., np.newaxis]

def norm(X):
    '''Computes a normalized version of an array of vectors.'''
    return X / mag1(X)

def plus(X):
    '''Return a shifted version of an array of vectors.'''
    return np.roll(X, -1, 0)

def minus(X):
    '''Return a shifted version of an array of vectors.'''
    return np.roll(X, +1, 0)

def cross(X, Y):
    '''Return the cross-product of two vectors.'''
    return np.cross(X, Y)

def proj(X, Y):
    r'''Return the projection of one vector onto another.
    
    Parameters
    ----------
    X, Y : vector array
    
    Returns
    -------
    Z : vector array
        :math:`\vec{Z} = \frac{\vec{X} \cdot \vec{Y}}{|Y|^2} \vec{Y}`
    '''
    Yp = norm(Y)
    return dot1(Yp, X) * Yp

def midpoint_delta(X):
    '''Returns center point and vector of each edge of the polygon defined by the points.'''
    Xp = plus(X)
    return (Xp + X) / 2., (Xp - X)

def arb_perp(V):
    '''For each vector, return an arbitrary vector that is perpendicular.
    
    **Note: arbitrary does not mean random!**'''
    p = eye(3, dtype=V.dtype)[argmin(V, -1)]
    return norm(p - proj(p, V))


def apply_basis(X, B):
    '''Transform each vector into the specified basis/bases.
    
    Parameters
    ----------
    X : vector array, shape [..., 3]
    B : orthonormal basis array, shape [..., 3, 3]
    
    Returns
    -------
    Y : vector array, shape [..., 3]
        X transformed into the basis given by B
    '''

    
    return (np.asarray(X)[..., np.newaxis, :] * B).sum(-1)
    


#------------------------------------------------------------------------------
# Building vectors intelligently
#------------------------------------------------------------------------------
def vec(x=[0], y=[0], z=[0]):
    '''Generate a [..., 3] vector from seperate x, y, z.

    Parameters
    ----------
    x, y, z: array
        coordinates; default to 0, may have any shape
        
    Returns
    -------
    X : [..., 3] array'''
    
    x, y, z = map(np.asarray, [x, y, z])
    
    s = [1]
    
    for a in (x, y, z):
        while a.ndim > len(s): s.prepend(1)
        s = [max(ss, n) for ss, n in zip(s, a.shape)]

    v = np.empty(s + [3], 'd')
    v[..., 0] = x
    v[..., 1] = y
    v[..., 2] = z
    
    return v

#------------------------------------------------------------------------------
# Rotations and basis operations
#------------------------------------------------------------------------------

def rot(a, X=None, cutoff=1E-10):
    '''Rotate points around an arbitrary axis.
    
    Parameters
    ----------
    a : [..., 3] array
        Rotation vector, will rotate counter-clockwise around axis by an amount
        given be the length of the vector (in radians).  May be a single vector
        or an array of vectors if each point is rotated separately.
        
    X : [..., 3] array
        Vectors to rotate; if not specified generates a rotation basis instead.
    
    cutoff : float
        If length of vector less than this value (1E-10 by default), no rotation
        is performed.  (Used to avoid basis errors)
    
    Returns
    -------
    Y : [..., 3] array
        Rotated vectors or rotation basis.
    '''
    
    #B = np.eye(3, dtype='d' if X is None else X.dtype)
    
    a = np.asarray(a)
    if X is None: X = np.eye(3).astype(a.dtype)

    phi = mag(a)    
    if phi.max() < 1E-10: return X
    
    #http://en.wikipedia.org/w/index.php?title=Rotation_matrix#Axis_and_angle
    n = norm(a)
    n[np.where(np.isnan(n).any(-1))] = (1, 0, 0)
    
    B = np.zeros(a.shape[:-1] + (3, 3), dtype=a.dtype)
    c = np.cos(phi)
    s = np.sin(phi)
    C = 1 - c
    
    for i in range(3):
        for j in range(3):
            if i == j:
                extra = c
            else:
                if (j - i)%3 == 2:
                    extra = +s * n[..., (j-1)%3]
                else:
                    extra = -s * n[..., (j+1)%3]
                                    
            B[..., i, j] = n[..., i]*n[..., j]*C + extra
    
    ##Create a new basis where the rotation is simply in x
    #Ba = normalize_basis(a[..., np.newaxis, :]) 
    #
    ###B = apply_basis(B, Ba) #This was pointless, B was 1
    ##y, z = B[..., 1, :].copy(), B[..., 2, :].copy()
    ##
    ##c, s = np.cos(phi), np.sin(phi)
    ##
    ###Rotate in new basis
    ##B[..., 1, :] = y*c - z*s
    ##B[..., 2, :] = y*s + z*c
    ##
    ##B = apply_basis(B, Ba.T).T
    #
    #B = np.zeros_like(Ba)
    #B[..., 0, 0] = 1
    #B[..., 1, 1] = +cos(phi)
    #B[..., 1, 2] = -sin(phi)
    #B[..., 2, 1] = +sin(phi)
    #B[..., 2, 2] = +cos(phi)
    #
    #B = apply_basis(B, Ba)
    
    if X is not None: return apply_basis(X, B)
    else: return B




def normalize_basis(B):
    '''Create right-handed orthonormal basis/bases from input basis.
    
    Parameters
    ----------
    B : [..., 1-3, 3] array
        input basis, should be at least 2d.  If the second to last axis has
        1 vectors, it will automatically create an arbitrary orthonormal basis
        with the specified first vector.
        (note: even if three bases are specified, the last is always ignored,
        and is generated by a cross product of the first two.)
    
    Returns
    -------
    NB : [..., 3, 3] array
        orthonormal basis
    '''
    
    B = np.asarray(B)
    NB = np.empty(B.shape[:-2] + (3, 3), dtype='d')
    
    
    v1 = norm(B[..., 0, :])
    v1[np.where(np.isnan(v1).any(-1))] = (1, 0, 0)
    
    
    v2 = B[..., 1, :] if B.shape[-2] >= 2 else np.eye(3)[np.argmin(abs(v1), axis=-1)]
    v2 = norm(v2 - v1 * dot1(v1, v2))
    v3 = cross(v1, v2)
    
    for i, v in enumerate([v1, v2, v3]): NB[..., i, :] = v
    
    return NB
    


if __name__ == '__main__':
    TEST = 2

    if TEST == 1:    
        x = np.eye(3)[0]
        a = vec(z = np.linspace(0, np.pi, 5))
        print a
        
        y = rot(a, x)
        print y

    if TEST == 2:
        from pylab import *
        
        NP = 10
        x = arange(NP) * 2 * pi / NP
        
        yd = [sin(x), cos(x), -sin(x), -cos(x)]
        
        
        for i in range(4):
            y = yd[i]
            yy = yd[0]
            yi = d5p(yy, h=x[1]-x[0], d=i, closed=True)
            plot(x, y+2*i)
            plot(x, yi+2*i, '.')
            print i, '%.12f' % mean(abs(y - yi))
            
        show()
