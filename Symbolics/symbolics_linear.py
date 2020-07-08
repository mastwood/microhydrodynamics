import sympy as sy
from sympy import Q
import sympy.diffgeom as diffg 
import numpy as np
import sympy.vector as sv 
from sympy.diffgeom.rn import R3_r
import matplotlib.pyplot as pl
import subprocess

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

G1=(3/4)*(r1r1/(rr1.norm()**2) + sy.eye(2)/(rr1.norm()))
G2=(3/4)*(r2r2/(rr2.norm()**2) + sy.eye(2)/(rr2.norm()))

u1 = G1*v1 
u2 = G2*v2 

G=sy.Matrix([[0,0],[0,0]])
G[:,0]=u1 
G[:,1]=u2 
_detG=sy.lambdify([y1,y2,x11,x22],sy.det(G))
detG=np.vectorize(_detG,excluded={0,1})

Xx1=np.arange(-10,10,0.5)
Xx2=np.arange(-10,10,0.5)
X1,X2=np.meshgrid(Xx1,Xx2)
fig, ax = pl.subplots()
counter=0
for i in np.arange(-5,5,0.05):
    pl.cla()
    Z=detG(i,i,X1,X2)
    CS = ax.contour(X1,X2,Z)
    ax.clabel(CS, inline=1, fontsize=10)
    ax.set_xlabel("x_1")
    ax.set_ylabel("x_2")
    pl.pause(0.05)
    #pl.savefig('lin_det_'+('%03d'%counter)+'.png')
    print(i)
#subprocess.call(['ffmpeg','-y', '-i', 'lin_det_%03d.png','-c:v', 'libx264', '-pix_fmt', 'yuv420p', 'out3.mp4'])
pl.show()

