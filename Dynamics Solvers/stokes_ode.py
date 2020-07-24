# Solves general trajectories given a control function

import time 
import numpy as np 
import matplotlib.pyplot as pl 
import scipy.integrate as ode
import matplotlib.animation as animation
import matplotlib.colors as colors
import subprocess
from StokesSingularities import *

def circ_control_position(theta1,theta2):
    x11=np.cos(theta1)
    x12=np.sin(theta1)
    x21=np.cos(theta2)
    x22=np.sin(theta2)
    x1=np.array([x11,x12])
    x2=np.array([x21,x22])
    return x1,x2 

def circ_control_velocity(u1,u2,theta1,theta2):
    x11=np.cos(theta1)
    x12=np.sin(theta1)
    x21=np.cos(theta2)
    x22=np.sin(theta2)
    x1=np.array([x11,x12])
    x2=np.array([x21,x22])
    v11=-u1*np.sin(theta1)
    v12=u1*np.cos(theta1)
    v21=-u2*np.sin(theta2)
    v22=u2*np.cos(theta2)
    v1=np.array([v11,v12])
    v2=np.array([v21,v22])
    return v1,v2 

def circ_velocity_field(u1,u2,theta1,theta2,y):
    y_velocity1=stokeslet_vec(y-circ_control_position(theta1,theta2)[0],circ_control_velocity(u1,u2,theta1,theta2)[0])
    #contribution from first active particle
    y_velocity2=stokeslet_vec(y-circ_control_position(theta1,theta2)[1],circ_control_velocity(u1,u2,theta1,theta2)[1])
    #contribution from second active particle
    return y_velocity1+y_velocity2
Circ_ODE = lambda t,y,args: circ_velocity_field(args[0],args[1],args[2],args[3],y)

def circ_solver(u1vals,u2vals,theta1vals,theta2vals,y0,tvals):
    dt=tvals[1]-tvals[0]
    trajectory=np.empty((tvals.size,3)) #trajectory of particle
    Solver=ode.ode(Circ_ODE).set_integrator('LSODE')
    Solver.set_initial_value(y0,tvals[0])
    print('Solving ODE...')
    for i in range(len(tvals)):
        Solver.set_f_params(u1vals[i],u2vals[i],theta1vals[i],theta2vals[i])
        trajectory[i,:]=np.array(Solver.integrate(Solver.t+dt))
        print('t = ' + str(tvals[i]))   
    return trajectory

def lin_control(t):
    return t
def lin_vel(t):
    return 1

def lin_velocity_field(y,x,u):
    y_velocity1=stokesletdipole_vec(y-np.array([x,0]),np.array([-u,0]),np.array([u,0]))
    return y_velocity1
Lin_ODE = lambda t,y: lin_velocity_field(y,lin_control(t),lin_vel(t))

def lin_solver(xvals,uvals,y0,tvals):
    dt=tvals[1]-tvals[0]
    trajectory=np.empty((tvals.size,2)) #trajectory of particle
    Solver=ode.ode(Lin_ODE).set_integrator('dopri5')
    Solver.set_initial_value(y0,tvals[0])
    print('Solving ODE...')
    for i in range(len(tvals)):
        trajectory[i,:]=np.array(Solver.integrate(Solver.t+dt))
        print('t = ' + str(tvals[i]))   
    return trajectory

def _filt(x):
    return np.tanh(x)
filt=np.vectorize(_filt)

tvalss=np.arange(-200,200,0.05)
xvals=lin_control(tvalss)
uvals=lin_vel(tvalss)
y0=np.array([0,2])

soln = lin_solver(xvals,uvals,y0,tvalss)

tvals=np.arange(-200,200,5)
# pl.plot(tvals,soln[:,0])
# pl.plot(tvals,soln[:,1])
# pl.show()
fig=pl.figure()
ax=fig.add_subplot(111)
counter=0
xv=np.linspace(-2,2,30)
yv=np.linspace(-1,3,30)

xx,yy=np.meshgrid(xv,yv)
ax.plot(soln[:,0],soln[:,1])
pl.grid()
# def animate(counter):
#     counter=int(counter*5/0.05)
#     ax.clear()
#     t=tvalss[counter]
#     ux=np.zeros_like(xx)
#     uy=np.zeros_like(yy)
#     uu=np.zeros_like(xx)
#     for i in range(xv.size):
#         for j in range(yv.size):
#             ux[i,j]=lin_velocity_field(np.array([xx[i,j],yy[i,j]]),lin_control(t),lin_vel(t))[0]
#             uy[i,j]=lin_velocity_field(np.array([xx[i,j],yy[i,j]]),lin_control(t),lin_vel(t))[1]
#             uu[i,j]=filt(np.abs(ux[i,j]**2+uy[i,j]**2))
#     Q=ax.quiver(xx,yy,ux,uy,uu,angles='xy',scale=20)
#     ax.set_xlim([-2,2])
#     ax.set_ylim([-1,3])
#     pl.grid()
#     ax.scatter(soln[counter,0],soln[counter,1],c='k',s=20)
#     ax.plot(soln[0:counter,0],soln[0:counter,1])
#     ax.scatter(xvals[counter],0,c='k',s=20)

#     counter=counter+1
# ani=animation.FuncAnimation(fig,animate,interval=1,frames=tvals.size)

#ani.save('Line_Long.mp4',fps=10,dpi=200)
pl.show()