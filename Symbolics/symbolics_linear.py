import sympy as sy
from sympy import Q
import sympy.diffgeom as diffg 
import numpy as np
import numpy.linalg as nplg
import matplotlib.pyplot as pl
import subprocess
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.optimize import fsolve
x11 = sy.symbols('x_11',real=True)
x22 = sy.symbols('x_22',real=True)
y1,y2 = sy.symbols('y_1 y_2',real=True)
v11=sy.symbols('v_11',real=True)
v22=sy.symbols('v_22',real=True)
x12=0
v12=0
v21=0
x21=0
v11=1
v22=1

r11=y1-x11;r12=y2-x12
r21=y1-x21;r22=y2-x22

r1=sy.Array([r11,r12])
r2=sy.Array([r21,r22])
rr1=sy.Matrix([r11,r12])
rr2=sy.Matrix([r21,r22])
v1=sy.Matrix([v11,v12])
v2=sy.Matrix([v21,v22])

r1r1 = sy.tensorproduct(r1,r1).tomatrix()
r2r2 = sy.tensorproduct(r2,r2).tomatrix()

G1=0.01*(3/4)*(r1r1/(rr1.norm()**3) + sy.eye(2)/(rr1.norm()))
G2=0.01*(3/4)*(r2r2/(rr2.norm()**3) + sy.eye(2)/(rr2.norm()))

u1 = G1*v1 
u2 = G2*v2 

G=sy.Matrix([[0,0],[0,0]])
G[:,0]=u1 
G[:,1]=u2 
# detG3=sy.det(G) 
# print(sy.latex(detG3))


def _detG(y1,y2,x1,x2):
    r1=np.matrix([y1-x1,y2])
    r2=np.matrix([y1,y2-x2])
    r1r1=np.outer(r1,r1)
    r2r2=np.outer(r2,r2)
    R1=nplg.norm(r1)
    R2=nplg.norm(r2)
    G1=(3/4)*(r1r1/(R1**3) + np.eye(2)/(R1))
    G2=(3/4)*(r2r2/(R2**3) + np.eye(2)/(R2))
    C=np.matmul(G1,np.matrix([[1,0],[0,0]]))+np.matmul(G2,np.matrix([[0,0],[0,1]]))
    return nplg.det(C)
detG=np.vectorize(_detG,excluded={0,1})

def _detG3(y1,y2,x1,x2):
    r1=np.array([y1-x1,y2])
    r2=np.array([y1,y2-x2])
    R1=nplg.norm(r1)
    R2=nplg.norm(r2)
    det = np.log((R2**(3/2)*r1[0]**2+R2)*(R1**(3/2)*r2[1]**2+R1)-r1[0]*r1[1]*r2[0]*r1[1])
    return det
detG3=np.vectorize(_detG3,excluded={0,1})


def _detG2(y1,x1,x2):
    return fsolve(lambda y2: ((x1-y1)**2+np.sqrt((y1-x2)**2+y2**2))*((x2-y2)**2+np.sqrt((y2-x2)**2+y1**2)-(y1-x1)*(y2-x2)*y1*y2),0)
detG2=np.vectorize(_detG2)
#_detG=sy.lambdify([y1,y2,x11,x22],sy.det(G))
#detG=np.vectorize(_detG,excluded={0,1})
Xx1=np.arange(-10,10,0.1)
Xx2=np.arange(-10,10,0.1)
Yy =np.arange(-5,5,0.1)

mode=1

if mode==0:
    X1,X2,Y1=np.meshgrid(Xx1,Xx2,Yy)
    fig=pl.figure()
    axes=fig.add_subplot(111,projection='3d')
    Z=detG2(Y1,X1,X2)
    axes.scatter(X1,X2,Y1,facecolors=cm.Oranges(Z))
    pl.plot()


    X1,X2=np.meshgrid(Xx1,Xx2)
    fig,ax=pl.subplot(1)



if mode==1:
    X1,X2=np.meshgrid(Xx1,Xx2)
    fig, axes = pl.subplots(5,5,sharex=True,sharey=True)
    fig.subplots_adjust(hspace=0,wspace=0)
    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

    counter=0
    for i in np.arange(-2,3,1):
        for j in np.arange(-2,3,1):
            a=4-(i+2)
            b=j+2
            axes[a,b].clear()

            Z=detG3(i,j,X1,X2)
            CS = axes[a,b].pcolormesh(X1,X2,Z)
            CS2 = axes[a,b].contour(X1,X2,Z,colors='w')

            #CS = axes[a,b].contour(X1,X2,Z,colors='r',linestyles='dotted',levels=[-0.01,0.01])

            axes[a,b].clabel(CS2, inline=1, fontsize=8)
            #pl.savefig('lin_det_'+('%03d'%counter)+'.png')
            print([i,j])

    axes[4,0].set_xlabel(r'$x_1 \in [-10,10],  y_1=-2$')
    axes[4,1].set_xlabel(r'$x_1 \in [-10,10],  y_1=-1$')
    axes[4,2].set_xlabel(r'$x_1 \in [-10,10],  y_1=0$')
    axes[4,3].set_xlabel(r'$x_1 \in [-10,10],  y_1=1$')
    axes[4,4].set_xlabel(r'$x_1 \in [-10,10],  y_1=2$')
    axes[0,0].set_ylabel(r'$x_2 \in [-5,5],  y_2=-2$')
    axes[1,0].set_ylabel(r'$x_2 \in [-5,5],  y_2=-1$')
    axes[2,0].set_ylabel(r'$x_2 \in [-5,5],  y_2=0$')
    axes[3,0].set_ylabel(r'$x_2 \in [-5,5],  y_2=1$')
    axes[4,0].set_ylabel(r'$x_2 \in [-5,5],  y_2=2$')

    fig.colorbar(CS,ax=axes.ravel().tolist())
    pl.show()

