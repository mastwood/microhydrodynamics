#Computes the determinant of the stokeslet.
import sympy as sy
from sympy import Q
import sympy.diffgeom as diffg 
import numpy as np
import sympy.vector as sv 
from sympy.diffgeom.rn import R3_r
import matplotlib.pyplot as pl

x1,x2,x3 = sy.symbols('r_1 r_2 r_3',real=True)
y1,y2,y3 = sy.symbols('y_1 y_2 y_3',real=True)
mu = sy.symbols('mu',real=True)
a=sy.symbols('a',real=True)

r = sy.Array([y1-x1,y2-x2,y3-x3])
rv=sy.Matrix([y1-x1,y2-x2,y3-x3])
R=rv.norm()
rr = sy.tensorproduct(r,r)
G= (3/4)*a*(rr.tomatrix()/(R**3)+sy.eye(3)/R)
print(sy.latex(sy.det(G)))