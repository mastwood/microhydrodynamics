# -*- coding: utf-8 -*-
"""
Quaternions

Created on Fri May 24 16:32:41 2019

@author: Henry Shum
"""

import numpy as np
#import chebyshev
#import numpy.polynomial.chebyshev as cheb


def quat2director(q, i):
    """
    Return a specific director from given quaternion.
    """

    assert q.shape == (4,), 'Quaternions must have 4 components'
    
    qnorm2inv = 1./sum(q**2)
    
    if i == 0:
        return qnorm2inv * np.array([q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2,
                                     2.*(q[1]*q[2] + q[0]*q[3]),
                                     2.*(q[1]*q[3] - q[0]*q[2])])
    elif i == 1:
        return qnorm2inv * np.array([2.*(q[1]*q[2] - q[0]*q[3]),
                                     q[0]**2 + q[2]**2 - q[1]**2 - q[3]**2,
                                     2.*(q[2]*q[3] + q[0]*q[1])])
    else:
        return qnorm2inv * np.array([2.*(q[1]*q[3] + q[0]*q[2]),
                                     2.*(q[2]*q[3] - q[0]*q[1]),
                                     q[0]**2 + q[3]**2 - q[1]**2 - q[2]**2])

            
def quat2mat(q):
    """
    Convert quaternion to 3x3 matrix whose columns are the corresponding
    basis vectors
    """

    assert q.shape == (4,), 'Quaternions must have 4 components'
    
    qnorm2inv = 1./sum(q**2)
    
    mat = qnorm2inv * np.array([[q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2,
                                2.*(q[1]*q[2] - q[0]*q[3]),
                                2.*(q[1]*q[3] + q[0]*q[2])],
                               [2.*(q[1]*q[2] + q[0]*q[3]),
                                q[0]**2 + q[2]**2 - q[1]**2 - q[3]**2,
                                2.*(q[2]*q[3] - q[0]*q[1])],
                               [2.*(q[1]*q[3] - q[0]*q[2]),
                                2.*(q[2]*q[3] + q[0]*q[1]),
                                q[0]**2 + q[3]**2 - q[1]**2 - q[2]**2]])

    return mat


def quatarray2vecarray(qarray):
    """
    Convert array of quaternions to three basis vector arrays
    qarray can be any shape but first dimension must have length 4.
    """
    
    qshape = qarray.shape
    assert qshape[0] == 4, 'Quaternions must have 4 components'
    
    if len(qshape) == 1: # Single quaternion input, not as array
        d1, d2, d3 = np.transpose(quat2mat(qarray))
        
    else:
        #assert len(qshape) == 2, 'Expected array of quaternions'
        qflat = qarray.reshape((4, -1))
        nq = qflat.shape[1]
    
        d1 = np.zeros([3, nq])
        d2 = np.zeros([3, nq])
        d3 = np.zeros([3, nq])
        for s in range(nq):
            d1[:,s], d2[:,s], d3[:,s] = np.transpose(quat2mat(qflat[:, s]))
        
        d1 = d1.reshape((3,) + qarray.shape[1:])
        d2 = d2.reshape((3,) + qarray.shape[1:])
        d3 = d3.reshape((3,) + qarray.shape[1:])
        
    return d1, d2, d3

def qdot_from_w(q, w, frame='body'):
    """
    Calculate qdot from angular velocity w.
    Default is for w to be expressed in body frame. Can also choose lab frame.
    """
    if frame == 'lab':
        R = quat2mat(q)
        w = np.matmul(w, R)
    elif frame == 'body':
        pass
    else:
        raise Exception('Unsupported frame: ' + frame)
        
    dq = np.zeros(4)
    dq[0] = 0.5 * (          - w[0]*q[1] - w[1]*q[2] - w[2]*q[3])
    dq[1] = 0.5 * (w[0]*q[0]             + w[2]*q[2] - w[1]*q[3])
    dq[2] = 0.5 * (w[1]*q[0] - w[2]*q[1]             + w[0]*q[3])
    dq[3] = 0.5 * (w[2]*q[0] + w[1]*q[1] - w[0]*q[2]            )
    
    return dq