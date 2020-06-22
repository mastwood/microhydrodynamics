import sympy as sy
import sympy.diffgeom as diffg 
import numpy as np
import sympy.vector as sv 
from sympy.diffgeom.rn import R3_r
x1,x2,x3 = sy.symbols('x_1 x_2 x_3')
y1,y2,y3 = sy.symbols('y_1 y_2 y_3')
mu, eta = sy.symbols('mu eta')
theta,phi, a=sy.symbols('theta, phi, a')

r1 = sy.Array([y1-a*sy.sin(theta)*sy.cos(phi),y2-a*sy.sin(theta)*sy.sin(phi),y3-a*sy.cos(phi)])
r1vec=sy.Matrix([y1-a*sy.sin(theta)*sy.cos(phi),y2-a*sy.sin(theta)*sy.sin(phi),y3-a*sy.cos(phi)])
R1=r1vec.norm()
rr1 = sy.tensorproduct(r1,r1)
G1= (3/4)*eta*(rr1.tomatrix()/(R1**2)+sy.eye(3)/R1)
#+(1/8)*(eta**3)*(2*sy.eye(3)/(R**3)-6*rr.tomatrix()/(R**5))

G10=G1.col(0) 
G11=G1.col(1) 
G12=G1.col(2)

uvec1= a*sy.cos(theta)*sy.cos(phi)*G10+a*sy.cos(theta)*sy.sin(phi)*G11 - a*sy.sin(theta)*G12
uvec2= a*sy.sin(theta)*sy.cos(phi)*G11 - a*sy.sin(theta)*sy.cos(phi)*G10

gmat=sy.Matrix([uvec1[0:2],uvec2[0:2]])
print(sy.det(gmat))