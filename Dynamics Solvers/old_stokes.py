# Implements the Green's function method for solving Stokes equation.

import numpy as np 
import scipy as sc
from scipy import integrate as si
import numpy.matlib as nm
import numpy.linalg as nl
import matplotlib.pyplot as pl
from scipy.interpolate import griddata

# Inputs:
# position -------------- r = [float, float, float]
# dynamic viscosity ----- mu = float 
def OseenTensor(r,mu):
    J0 = 1/(8*np.pi*mu)
    J = J0*(nm.identity(3)/nl.norm(r,ord=2) + np.outer(r,r)/(nl.norm(r,ord=2)**3))
    return J

# Inputs:
# forcing ----------------- force = float(function([float,float,float]))
# position ---------------- pos = [float,float,float]
# args -------------------- args = [dynamic_viscosity=float, fundamental_solution=string]
# * argument options:
# * fundamental_solution = "stokeslet", "stresslet"
def SolveStokes(force,pos,args):
    mu=args[0] 
    if args[1]=="stokeslet":
        res = si.quad_vec(lambda x: nm.einsum("ij,j", OseenTensor([pos[0]-x,pos[1],pos[2]],mu),f(x,0,0)), -np.inf,np.inf)[0]
    return res

#forcing function
def f(x,y,z):
    return [0.0001*np.exp(-x**2),0,0]

#helper function for solver
def u(x,y): 
    #return np.einsum("ij,j",OseenTensor([x,y,0],1).tolist(),[1,0,0])
    return SolveStokes(f,np.array([x,y,0]),[1,"stokeslet"])

#coordinates
x=np.linspace(-5,5,10)
y=x

#solve on grid
U=np.empty((len(x),len(y)))
V=np.empty((len(x),len(y)))
for i in range(len(x)):
    for j in range(len(y)):
        print(u(x[i],y[j]))
        U[i,j]=u(x[i],y[j])[0]
        V[i,j]=u(x[i],y[j])[1]

X,Y = np.meshgrid(x,y)
U.shape=V.shape=(len(x),len(y))
pl.streamplot(X, Y, np.array(U), np.array(V), linewidth=1)
pl.show()