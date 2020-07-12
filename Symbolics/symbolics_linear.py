import sympy as sy
from sympy import Q
import sympy.diffgeom as diffg 
import numpy as np
import numpy.linalg as nplg
import matplotlib.pyplot as pl
import subprocess
import matplotlib

# x11 = sy.symbols('x_11',real=True)
# x22 = sy.symbols('x_22',real=True)
# y1,y2 = sy.symbols('y_1 y_2',real=True)
# v11=sy.symbols('v_11',real=True)
# v22=sy.symbols('v_22',real=True)
# x12=0
# v12=0
# v21=0
# x21=0
# v11=1
# v22=1

# r11=y1-x11;r12=y2-x12
# r21=y1-x21;r22=y2-x22

# r1=sy.Array([r11,r12])
# r2=sy.Array([r21,r22])
# rr1=sy.Matrix([r11,r12])
# rr2=sy.Matrix([r21,r22])
# v1=sy.Matrix([v11,v12])
# v2=sy.Matrix([v21,v22])

# r1r1 = sy.tensorproduct(r1,r1).tomatrix()
# r2r2 = sy.tensorproduct(r2,r2).tomatrix()

# G1=0.01*(3/4)*(r1r1/(rr1.norm()**2) + sy.eye(2)/(rr1.norm()))
# G2=0.01*(3/4)*(r2r2/(rr2.norm()**2) + sy.eye(2)/(rr2.norm()))

# u1 = G1*v1 
# u2 = G2*v2 

# G=sy.Matrix([[0,0],[0,0]])
# G[:,0]=u1 
# G[:,1]=u2 

def _detG(y1,y2,x1,x2):
    r1=np.matrix([y1-x1,y2])
    r2=np.matrix([y1,y2-x2])
    r1r1=np.outer(r1,r1)
    r2r2=np.outer(r2,r2)
    R1=nplg.norm(r1)
    R2=nplg.norm(r2)
    G1=(3/4)*(r1r1/(R1**2) + np.eye(2)/(R1))
    G2=(3/4)*(r2r2/(R2**2) + np.eye(2)/(R2))
    C=np.matmul(G1,np.matrix([[1,0],[0,0]]))+np.matmul(G2,np.matrix([[0,0],[0,1]]))
    return nplg.det(C)
detG=np.vectorize(_detG,excluded={0,1})


#_detG=sy.lambdify([y1,y2,x11,x22],sy.det(G))
#detG=np.vectorize(_detG,excluded={0,1})
Xx1=np.arange(-5,5,0.1)
Xx2=np.arange(-5,5,0.1)
X1,X2=np.meshgrid(Xx1,Xx2)
fig, axes = pl.subplots(3,3,sharex=True,sharey=True)
fig.subplots_adjust(hspace=0,wspace=0)
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

counter=0
for i in np.arange(1,4,1):
    for j in np.arange(1,4,1):
        a=3-i
        b=j-1
        axes[a,b].clear()

        Z=detG(i,j,X1,X2)
        CS = axes[a,b].pcolormesh(X1,X2,Z,vmin=0,vmax=1)
        CS2 = axes[a,b].contour(X1,X2,Z,colors='w')

        #CS = axes[a,b].contour(X1,X2,Z,colors='r',linestyles='dotted',levels=[-0.01,0.01])

        axes[a,b].clabel(CS2, inline=1, fontsize=10)
        #pl.savefig('lin_det_'+('%03d'%counter)+'.png')
        print([i,j])

axes[2,0].set_xlabel(r'$x_1 \in [-5,5],  y_1=1$')
axes[2,1].set_xlabel(r'$x_1 \in [-5,5],  y_1=2$')
axes[2,2].set_xlabel(r'$x_1 \in [-5,5],  y_1=3$')
axes[0,0].set_ylabel(r'$x_2 \in [-5,5],  y_2=1$')
axes[1,0].set_ylabel(r'$x_2 \in [-5,5],  y_2=2$')
axes[2,0].set_ylabel(r'$x_2 \in [-5,5],  y_2=3$')

fig.colorbar(CS,ax=axes.ravel().tolist())
pl.show()

