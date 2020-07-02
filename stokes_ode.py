# Solves general trajectories given a control function

import time 
import numpy as np 
import matplotlib.pyplot as pl 
import scipy.integrate as odes
import subprocess
from scipy.spatial.transform import Rotation as Rrr


def stokeslet_vec(x, e):
    # Spagnolie & Lauga 2012 notation
    rinvsq = 1.0/np.dot(x, x)
    rinv = 1.0/np.linalg.norm(x)
    
    vec = rinv*(e + np.dot(x,e)*x*rinvsq)
    
    return vec

def circ_control_position(theta1,theta2):
    x11=np.cos(theta1)
    x12=np.sin(theta1)
    x21=np.cos(theta2)
    x22=np.sin(theta2)
    x1=np.array([x11,x12])
    x2=np.array([x21,x22])
    return x1,x2 

def circ_control_velocity(u1,u2,theta1,theta2)
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

def circ_solver(u1vals,u2vals,theta1vals,theta2vals,y0,tvals)
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


