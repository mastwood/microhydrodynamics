from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as pl
import subprocess
import time 
import scipy.integrate as odes
import subprocess
from scipy.spatial.transform import Rotation as Rrr

g=GEKKO(remote=True)

nt = 201 #number of timesteps
g.time=np.linspace(0,25,nt)

#STATUS = 1 implies that the variable is being optimized by the solver
#DCOST is the amount which is added to the cost function when the variable is modified -> this prevents blowup
u1=g.MV(value=0); u1.STATUS=1; #These are the control variables.
u1.LOWER=-100
u1.UPPER=100
u2=g.MV(value=0); u2.STATUS=1; 
u2.LOWER=-100
u2.UPPER=100

# These are the coordinates of the passive particle
y1=g.CV(value=75)
y1.LOWER=50
y1.UPPER=100
y2=g.CV(value=75)
y2.LOWER=50
y2.UPPER=100

#Angular coordinates of the active particle
theta1=g.Var(value=0)
theta2=g.Var(value=np.pi/2)

v11=-u1*g.sin(theta1)
v12=u1*g.cos(theta1)
v21=-u2*g.sin(theta2)
v22=u1*g.cos(theta2)

x11=g.cos(theta1)
x12=g.sin(theta1)
x21=g.cos(theta2)
x22=g.sin(theta2)

r11=y1-x11
r12=y2-x12
r21=y1-x21
r22=y2-x22

#Constraint equations which relate the active particles' velocity to its' position
g.Equation(theta1.dt()==u1)
g.Equation(theta2.dt()==u1)

#ydot=6*pi*(v1+dot(y-x1,v1)*(y-x1)/dot(y-x1,y-x1))/sqrt(dot(y-x1,y-x1)) \
#    + 6*pi*(v2+dot(y-x2,v2)*(y-x2)/dot(y-x2,y-x2))/sqrt(dot(y-x2,y-x2)) 

#Function which computes the velocity field 
def stokeslet(y1,y2,theta1,theta2,u1,u2):
    k=6*np.pi 
    ydot1= (v11+(r11*v11+r12*v12)*r11/(r11**2+r12**2))/g.sqrt(r11**2+r12**2)+\
           (v21+(r21*v21+r22*v22)*r21/(r21**2+r22**2))/g.sqrt(r21**2+r22**2)
    ydot2= (v12+(r11*v11+r12*v12)*r12/(r11**2+r12**2))/g.sqrt(r11**2+r12**2)+\
           (v22+(r21*v21+r22*v22)*r22/(r21**2+r22**2))/g.sqrt(r21**2+r22**2)
    return 1*k*np.array([ydot1,ydot2])/(8*np.pi)

# Dynamical constraints
g.Equation(y1.dt()==stokeslet(y1,y2,theta1,theta2,u1,u2)[0])
g.Equation(y2.dt()==stokeslet(y1,y2,theta1,theta2,u1,u2)[1])

# Cost function
J=g.Var(value=0)

g.Equation(J.dt()==u1**2+u2**2)

final = g.Param(np.zeros(nt)); final[-1]=1
g.Minimize(J*final)

g.Minimize(final*1e5*(y1-90)**2)
g.Minimize(final*1e5*(y2-90)**2)

g.options.IMODE = 6  # optimal control
g.options.NODES = 4  # collocation nodes
g.options.SOLVER = 3 # solver
g.options.MAX_ITER = 2000

g.options.COLDSTART = 1
g.solve(disp=True) # Solve

g.options.COLDSTART = 0
g.options.TIME_SHIFT = 0
g.solve(disp=True) # Solve

fig,(ax1,ax2)=pl.subplots(2)
ax1.plot(g.time,y1.value,label='y1')
ax1.plot(g.time,y2.value,label='y2')
ax1.legend()
ax2.step(g.time,u1.value,label='u1')
ax2.step(g.time,u2.value,label='u2')
ax2.legend()
pl.savefig("optimal_control.png")

x11=np.cos(theta1.value)
x12=np.sin(theta1.value)
x21=np.cos(theta2.value)
x22=np.sin(theta2.value)

counter=0
while counter < len(g.time):
    fig = pl.figure()
    ax = fig.add_subplot(111)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.plot(y1.value[0:counter],y2.value[0:counter],lw=0.5, label='Passive')
    ax.plot(x11[0:counter],x12[0:counter],lw=0.5,label='Active 1')
    ax.plot(x21[0:counter],x22[0:counter],lw=0.5,label='Active 2')
    ax.scatter(y1.value[counter-1:counter],y2.value[counter-1:counter],c='b')
    ax.scatter(x11[counter-1:counter],x12[counter-1:counter],c='r')
    ax.scatter(x21[counter-1:counter],x22[counter-1:counter],c='r')

    pl.savefig('Movies\\imgopt_'+str("%03d"%(counter))+'.png')
    pl.close()
    print(counter)
    counter=counter+1
subprocess.call(['ffmpeg','-y', '-i', 'Movies\\imgopt_%03d.png','-c:v', 'libx264', '-pix_fmt', 'yuv420p', 'opt.mp4'])



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
    y_velocity1=6*np.pi*stokeslet_vec(y-circ_control_position(theta1,theta2)[0],circ_control_velocity(u1,u2,theta1,theta2)[0])
    #contribution from first active particle
    y_velocity2=6*np.pi*stokeslet_vec(y-circ_control_position(theta1,theta2)[1],circ_control_velocity(u1,u2,theta1,theta2)[1])
    #contribution from second active particle
    return 1*(y_velocity1+y_velocity2)/(8*np.pi)
Circ_ODE = lambda t,y,args: circ_velocity_field(args[0],args[1],args[2],args[3],y)

def circ_solver(u1vals,u2vals,theta1vals,theta2vals,y0,tvals):
    dt=tvals[1]-tvals[0]
    trajectory=np.empty((tvals.size,2)) #trajectory of particle
    Solver=odes.ode(Circ_ODE).set_integrator('DOP853',max_step=dt)
    Solver.set_initial_value(y0,tvals[0])

    print('Solving ODE...')
    for i in range(len(tvals)):
        Solver.set_f_params([u1vals[i],u2vals[i],theta1vals[i],theta2vals[i]])
        trajectory[i,:]=np.array(Solver.integrate(Solver.t+dt))   
    return trajectory

fig2 = pl.figure()
trajectory=circ_solver(u1.value,u2.value,theta1.value,theta2.value,[75,75],g.time)
pl.plot(trajectory[:,0],trajectory[:,1], label = 'Actual Trajectory')
pl.plot(y1.value,y2.value,label='Predicted Trajectory')
pl.legend()
pl.show()
