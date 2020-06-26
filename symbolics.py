import sympy as sy
from sympy import Q
import sympy.diffgeom as diffg 
import numpy as np
import sympy.vector as sv 
from sympy.diffgeom.rn import R3_r
import matplotlib.pyplot as pl

x1,x2,x3 = sy.symbols('x_1 x_2 x_3',real=True)
y1,y2,y3 = sy.symbols('y_1 y_2 y_3',real=True)
mu, eta = sy.symbols('mu eta',real=True)
theta_1,theta_2,theta_3, a=sy.symbols('theta_1,theta_2,theta_3, a',real=True)

r1 = sy.Array([y1-a*sy.cos(theta_1),y2-a*sy.sin(theta_1),0])
r1vec=sy.Matrix([y1-a*sy.cos(theta_1),y2-a*sy.sin(theta_1),0])

r2 = sy.Array([y1-a*sy.cos(theta_2),y2-a*sy.sin(theta_2),0])
r2vec=sy.Matrix([y1-a*sy.cos(theta_2),y2-a*sy.sin(theta_2),0])

r3 = sy.Array([y1-a*sy.cos(theta_3),y2-a*sy.sin(theta_3),0])
r3vec=sy.Matrix([y1-a*sy.cos(theta_3),y2-a*sy.sin(theta_3),0])

R1=r1vec.norm()
rr1 = sy.tensorproduct(r1,r1)

G1= (3/4)*eta*(rr1.tomatrix()/(R1**2)+sy.eye(3)/R1)
# +(1/8)*(eta**3)*(2*sy.eye(3)/(R**3)-6*rr.tomatrix()/(R**5))

R2=r2vec.norm()
rr2 = sy.tensorproduct(r2,r2)

G2= (3/4)*eta*(rr2.tomatrix()/(R2**2)+sy.eye(3)/R2)
# +(1/8)*(eta**3)*(2*sy.eye(3)/(R**3)-6*rr.tomatrix()/(R**5))

R3=r3vec.norm()
rr3 = sy.tensorproduct(r3,r3)

G3= (3/4)*eta*(rr3.tomatrix()/(R3**2)+sy.eye(3)/R3)
# +(1/8)*(eta**3)*(2*sy.eye(3)/(R**3)-6*rr.tomatrix()/(R**5))

G10=G1.col(0) 
G11=G1.col(1) 
G12=G1.col(2)

G20=G2.col(0) 
G21=G2.col(1) 
G22=G2.col(2)

G30=G3.col(0) 
G31=G3.col(1) 
G32=G3.col(2)

uvec1= a*sy.cos(theta_1)*G11-a*sy.sin(theta_1)*G10
uvec2= a*sy.cos(theta_2)*G21-a*sy.sin(theta_2)*G20

G = sy.Matrix([[0,0],[0,0]])
G[:,0]=uvec1[0:2]
G[:,1]=uvec2[0:2]

#G=G.col_insert(2,uvec3)

#print(sy.latex(G))
#print(sy.latex(sy.det(G)))

Y1=5
Y2=5
TH1=np.arange(-np.pi,np.pi,0.1)
TH2=np.arange(-np.pi,np.pi,0.1)
T1,T2=np.meshgrid(TH1,TH2)
A=1
Eta=1
Gfunc = np.vectorize(lambda theta1,theta2,yy1,yy2: sy.det(G.evalf(subs={y1:yy1,y2:yy2,theta_1:theta1,theta_2:theta2,a:A,eta:Eta})),excluded={y1,y2})

for i in np.arange(-100,100,10):
    Z=Gfunc(T1,T2,i,Y2)

    fig, ax = pl.subplots()
    CS = ax.contour(T1,T2,Z)
    ax.clabel(CS, inline=1, fontsize=10)
    ax.set_xlabel("theta_1")
    ax.set_ylabel("theta_2")
    pl.savefig("determinant_"+str(i)+".png")
    pl.close()
    print(i)