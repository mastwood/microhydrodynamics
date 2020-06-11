# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 18:06:15 2019

@author: Henry Shum
"""

import numpy as np

def stokeslet_vec(x, e):
    # Spagnolie & Lauga 2012 notation
    rinvsq = 1.0/np.dot(x, x)
    rinv = 1.0/np.linalg.norm(x)
    
    vec = rinv*(e + np.dot(x,e)*x*rinvsq)
    
    return vec

def stokesletdipole_vec(x, d, e):
    # Spagnolie & Lauga 2012 notation
    rinvsq = 1.0/np.dot(x, x)
    rinv = 1.0/np.linalg.norm(x)
    
    vec = rinvsq*(rinv*(np.dot(d,x)*e - np.dot(e,x)*d - np.dot(d,e)*x) +
                  3.0*np.dot(e,x)*np.dot(d,x)*x*rinv*rinvsq)
    
    return vec
    
def stokesletquadrupole_vec(x, c, d, e):
    # Spagnolie & Lauga 2012 notation
    rinvsq = 1.0/np.dot(x, x)
    rinv = 1.0/np.linalg.norm(x)
    
    g1 = (np.dot(d,e)*np.dot(c,x) + np.dot(c,e)*np.dot(d,x) + 
          np.dot(c,d)*np.dot(e,x))*x +\
         np.dot(d,x)*np.dot(e,x)*c + np.dot(c,x)*np.dot(e,x)*d - \
         np.dot(c,x)*np.dot(d,x)*e
    vec = rinv*rinvsq*(np.dot(d,e)*c + np.dot(c,e)*d - np.dot(c,d)*e -
                  3.0*rinvsq*g1 + 
                  15.0*rinvsq**2*np.dot(c,x)*np.dot(d,x)*np.dot(e,x)*x)
    
    return vec

def source_vec(x):
    # Spagnolie & Lauga 2012 notation
    rinvsq = 1.0/np.dot(x, x)
    rinv = 1.0/np.linalg.norm(x)
    
    vec = rinv*rinvsq*x
    
    return vec

def sourcedoublet_vec(x, e):
    # Spagnolie & Lauga 2012 notation
    rinvsq = 1.0/np.dot(x, x)
    rinv = 1.0/np.linalg.norm(x)

    vec = rinv*rinvsq*(-e + 3.0*np.dot(x,e)*x*rinvsq)
    
    return vec

def sourcequad_vec(x, d, e):
    # Spagnolie & Lauga 2012 notation
    rinvsq = 1.0/np.dot(x, x)
    rinv = 1.0/np.linalg.norm(x)
    
    vec = -3.0*rinvsq**2*rinv*(np.dot(d,x)*e+np.dot(e,x)*d+np.dot(d,e)*x -
                               5.0*np.dot(e,x)*np.dot(d,x)*x*rinvsq)
    
    return vec

def stresslet_vec(x, d, e):
    # Spagnolie & Lauga 2012 notation
    rinvsq = 1.0/np.dot(x, x)
    rinv = 1.0/np.linalg.norm(x)
    
    vec = -6.0*rinvsq**2*rinv*np.dot(d,x)*np.dot(e,x)*x
    
    return vec

def stressletim_vec(x, d, e, h):
    # Spagnolie & Lauga 2012 notation
    # No slip plane boundary at z=0. h is z co-ordinate of stresslet
    # x is vector from image point to observation point
    
    hsq = h**2
    
    xhat = np.array([1,0,0])
    yhat = np.array([0,1,0])
    zhat = np.array([0,0,1])
    
    vecx = -2.0*d[0]*source_vec(x) + \
            4.0*h*d[2]*sourcedoublet_vec(x, xhat) +\
            4.0*h*d[0]*sourcedoublet_vec(x, zhat) +\
            2.0*d[0]*stokesletdipole_vec(x, xhat, xhat) +\
            (-4.0)*d[0]*stokesletdipole_vec(x, zhat, zhat) +\
            d[1]*(stokesletdipole_vec(x, xhat, yhat) + 
                  stokesletdipole_vec(x, yhat, xhat)) -\
            d[2]*(stokesletdipole_vec(x, xhat, zhat) + 
                  stokesletdipole_vec(x, zhat, xhat)) +\
            4.0*hsq*(d[0]*sourcequad_vec(x, xhat, xhat) +\
                     d[1]*sourcequad_vec(x, xhat, yhat) -\
                     d[2]*sourcequad_vec(x, xhat, zhat)) +\
            4.0*h*(d[2]*stokesletquadrupole_vec(x,xhat,zhat,zhat) -\
                   d[0]*stokesletquadrupole_vec(x,xhat,xhat,zhat) -\
                   d[1]*stokesletquadrupole_vec(x,xhat,yhat,zhat))
            
    vecy = -2.0*d[1]*source_vec(x) + \
            4.0*h*d[2]*sourcedoublet_vec(x, yhat) +\
            4.0*h*d[1]*sourcedoublet_vec(x, zhat) +\
            2.0*d[1]*stokesletdipole_vec(x, yhat, yhat) +\
            (-4.0)*d[1]*stokesletdipole_vec(x, zhat, zhat) +\
            d[0]*(stokesletdipole_vec(x, xhat, yhat) + 
                  stokesletdipole_vec(x, yhat, xhat)) -\
            d[2]*(stokesletdipole_vec(x, yhat, zhat) + 
                  stokesletdipole_vec(x, zhat, yhat)) +\
            4.0*hsq*(d[1]*sourcequad_vec(x, yhat, yhat) +\
                     d[0]*sourcequad_vec(x, xhat, yhat) -\
                     d[2]*sourcequad_vec(x, yhat, zhat)) +\
            4.0*h*(d[2]*stokesletquadrupole_vec(x,yhat,zhat,zhat) -\
                   d[1]*stokesletquadrupole_vec(x,yhat,yhat,zhat) -\
                   d[0]*stokesletquadrupole_vec(x,xhat,yhat,zhat))
            
    vecz = -2.0*d[2]*source_vec(x) + \
            4.0*h*(d[0]*sourcedoublet_vec(x, xhat) +
                   d[1]*sourcedoublet_vec(x, yhat) -
                   d[2]*sourcedoublet_vec(x, zhat)) -\
            2.0*d[2]*stokesletdipole_vec(x, zhat, zhat) -\
            d[0]*(stokesletdipole_vec(x, xhat, zhat) + 
                  stokesletdipole_vec(x, zhat, xhat)) -\
            d[1]*(stokesletdipole_vec(x, yhat, zhat) + 
                  stokesletdipole_vec(x, zhat, yhat)) +\
            4.0*hsq*(d[2]*sourcequad_vec(x, zhat, zhat) -\
                     d[0]*sourcequad_vec(x, xhat, zhat) -\
                     d[1]*sourcequad_vec(x, yhat, zhat)) +\
            4.0*h*(d[0]*stokesletquadrupole_vec(x,xhat,zhat,zhat) +\
                   d[1]*stokesletquadrupole_vec(x,yhat,zhat,zhat) -\
                   d[2]*stokesletquadrupole_vec(x,zhat,zhat,zhat))
    
    vec = vecx*e[0] + vecy*e[1] + vecz*e[2]
    
    return vec

def stokeslet_tens(x0,x):
     r = x - x0
     
     rsq = np.dot(r, r)
     
     mat = np.eye(3)/np.sqrt(rsq) + np.multiply.outer(r,r)*rsq**(-3/2)
     
     return mat
 
    
def stresslet_tens(x0,x):
     r = x - x0
     
     rsq = np.dot(r, r)
     
     mat = -6*np.multiply.outer(r,np.multiply.outer(r,r))*rsq**(-5/2)
     
     return mat