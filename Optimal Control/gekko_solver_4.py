from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.colors as colors
import subprocess
import time 
import scipy.integrate as odes
from StokesSingularities import *

#This iteration includes interaction forces

IV=[50,50] #Initial position

g=GEKKO(remote=True)

nt = 400 #number of timesteps
g.time=np.linspace(0,100,nt)

#STATUS = 1 implies that the variable is being optimized by the solver
#DCOST is the amount which is added to the cost function when the variable is modified -> this prevents blowup

# These are the coordinates of the passive particle
y1=g.CV(value=IV[0])
y1.LOWER=-100
y1.UPPER=100
y2=g.CV(value=IV[1])
y2.LOWER=-100
y2.UPPER=100

x11=g.Var(value=2,lb=1,ub=3)
x12=g.Var(value=0,lb=-1,ub=1)
x21=g.Var(value=0,lb=-1,ub=1)
x22=g.Var(value=0,lb=-1,ub=1)

v11=g.MV(value=0);v11.STATUS=0
v12=g.MV(value=0);v12.STATUS=1
v21=g.MV(value=0);v21.STATUS=1
v22=g.MV(value=0);v22.STATUS=0

v11.LOWER=-10
v11.UPPER=10
v12.LOWER=-10
v12.UPPER=10
v21.LOWER=-10
v21.UPPER=10
v22.LOWER=-10
v22.UPPER=10

g.Equation(x11.dt()==v11)
g.Equation(x12.dt()==v12)
g.Equation(x21.dt()==v21)
g.Equation(x22.dt()==v22)

def K1(r1,r2,v1,v2):
    a=0.1
    coeff = a*(3/2)*((r1**2+r2**2)**0.5)*(8*(r1**2+r2**2)**2-9*(a**2)*(5*r1**2+6*r1*r2+5*r2**2))**(-1)
    m11=-4*((r1**2+r2**2)**2)*(2*r1**2+r2**2)+9*(a**2)*(r1**2+r1*r2+2*r2**2)
    m12=-4*r1*r2*((r1**2+r2**2)**2)-9*(a**2)*(r1**2+r1*r2+2*r2**2)
    m21=-4*r1*r2*((r1**2+r2**2)**2)-9*(a**2)*(r2**2+r1*r2+2*r1**2)
    m22=-4*((r2**2+r1**2)**2)*(2*r2**2+r1**2)+9*(a**2)*(r2**2+r1*r2+2*r1**2)
    return np.array([coeff*m11*v1+coeff*m12*v2,coeff*m21*v1+coeff*m22*v2])
def K2(r1,r2,v1,v2):
    a=0.1
    coeff = (9*(a**2)*(5*r1**2+6*r1*r2+5*r2**2)-8*((r1**2+r2**2)**2))**(-1)
    m11=8*(r1**2+r2**2)**2-9*(a**2)*(r1**2+3*r1*r2+4*r2**2)
    m12=9*(a**2)*(r1**2+3*r1*r2+4*r2**2)
    m21=9*(a**2)*(r2**2+3*r1*r2+4*r1**2)
    m22=8*(r1**2+r2**2)**2-9*(a**2)*(r2**2+3*r1*r2+4*r1**2)
    return np.array([coeff*m11*v1+coeff*m12*v2,coeff*m21*v1+coeff*m22*v2])

f1 = K1(x11-x21,x12-x22,v11,v12)-K2(x11-x21,x12-x22,v21,v22)
f2 = K1(x11-x21,x12-x22,v21,v22)-K2(x11-x21,x12-x22,v11,v12)

# Stokeslet fundamental solution.
def stokeslet(r1,r2,v1,v2):
    rinvsq1=1/(r1**2+r2**2)
    rinv1=rinvsq1**0.5
    rv1=r1*v1+r2*v2 
    ydot1=rinv1*(v1+rv1*r1*rinvsq1)
    ydot2=rinv1*(v2+rv1*r2*rinvsq1)
    return 0.1*np.array([ydot1,ydot2])*3/4

# Dynamical constraints

g.Equation(y1.dt()==stokeslet(y1/((y1**2+y2**2)**0.5),y2/((y1**2+y2**2)**0.5),f1[0]+f2[0],f1[1]+f2[1])[0])
g.Equation(y2.dt()==stokeslet(y1/((y1**2+y2**2)**0.5),y2/((y1**2+y2**2)**0.5),f2[0]+f1[0],f2[1]+f1[1])[1])

# Cost function
J=g.Var(value=0)

g.Equation(J==g.integral(g.abs(v11*f1[0]+v12*f1[1]+v21*f2[0]+v22*f2[1])))

final = g.Param(np.zeros(nt)); final[-1]=1
g.Minimize(J*final)
#g.Equation(1e3*g.exp(-((y1-x11)**2+(y2-x22)**2+(y1-x21)**2+(y2-x22)**2)) < 1)
g.Minimize(final*1e5*(y1-51)**2)
g.Minimize(final*1e5*(y2-52)**2)

g.options.IMODE = 6  # optimal control
g.options.NODES = 2  # collocation nodes
g.options.SOLVER = 3 # solver
g.options.MAX_ITER = 10000
g.solver_options={'print_info_string yes'}

g.options.COLDSTART = 1
g.solve(disp=True) # Solve

g.options.COLDSTART = 0
g.options.TIME_SHIFT = 0
g.solve(disp=True) # Solve

fig,(ax1,ax2)=pl.subplots(2)
ax1.plot(g.time,y1.value,label='y1')
ax1.plot(g.time,y2.value,label='y2')
ax1.legend()
ax2.plot(g.time,v11.value,label='v11')
ax2.plot(g.time,v12.value,label='v12')
ax2.plot(g.time,v21.value,label='v21')
ax2.plot(g.time,v22.value,label='v22')
ax2.legend()
pl.savefig("optimal_control.png")
#pl.show()
fig2,ax3=pl.subplots(1)
ax3.plot(y1.value,y2.value,label='trajectory')
pl.show()