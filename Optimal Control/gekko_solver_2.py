from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as pl
import subprocess
import time 
import scipy.integrate as odes
import subprocess
from scipy.spatial.transform import Rotation as Rrr

g=GEKKO(remote=True)

nt =501 #number of timesteps
g.time=np.linspace(0,100,nt)

#STATUS = 1 implies that the variable is being optimized by the solver
#DCOST is the amount which is added to the cost function when the variable is modified -> this prevents blowup

# These are the coordinates of the passive particle
y1=g.CV(value=-30)
y1.LOWER=-100
y1.UPPER=100
y2=g.CV(value=-30)
y2.LOWER=-100
y2.UPPER=100

x1=g.Var(value=1)
x1.LOWER=-100
x1.UPPER=100
x2=g.Var(value=0)
x2.LOWER=-100
x2.UPPER=100
v1=g.MV(value=0);v1.STATUS=1
v2=g.MV(value=0);v2.STATUS=1

def stokeslet(y1,y2,x1,x2,v1,v2):
    k=6*np.pi 
    r11=y1-x1
    r12=y2
    r21=y1
    r22=y2-x2

    ydot1= (v1+(r11*v1)*r11/(r11**2+r12**2))/g.sqrt(r11**2+r12**2)+\
           ((r22*v2)*r21/(r21**2+r22**2))/g.sqrt(r21**2+r22**2)
    ydot2= ((r11*v1)*r12/(r11**2+r12**2))/g.sqrt(r11**2+r12**2)+\
           (v2+(r22*v2)*r22/(r21**2+r22**2))/g.sqrt(r21**2+r22**2)
    return 1*k*np.array([ydot1,ydot2])/(8*np.pi)

# Dynamical constraints
g.Equation(x1.dt()==v1)
g.Equation(x2.dt()==v2)
g.Equation(y1.dt()==stokeslet(y1,y2,x1,x2,v1,v2)[0])
g.Equation(y2.dt()==stokeslet(y1,y2,x1,x2,v1,v2)[1])

# Cost function
J=g.Var(value=0)

g.Equation(J.dt()==v1**2+v2**2)

final = g.Param(np.zeros(nt)); final[-1]=1
g.Minimize(J*final)

g.Minimize(final*1e5*(y1-30)**2)
g.Minimize(final*1e5*(y2-30)**2)

g.options.IMODE = 6  # optimal control
g.options.NODES = 3  # collocation nodes
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
ax2.step(g.time,v1.value,label='v1')
ax2.step(g.time,v2.value,label='v2')
ax2.legend()
pl.savefig("optimal_control.png")

counter=0
while counter < len(g.time):
    fig = pl.figure()
    ax = fig.add_subplot(111)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.plot(y1.value[0:counter],y2.value[0:counter],lw=0.5, label='Passive')
    ax.scatter(y1.value[counter],y2.value[counter],c='b')
    ax.scatter(x1.value[counter],[0],c='r')
    ax.scatter([0],x2.value[counter],c='r')

    pl.savefig('Movies\\imgopt2_'+str("%03d"%(counter))+'.png')
    pl.close()
    print(counter)
    counter=counter+1
subprocess.call(['ffmpeg','-y', '-i', 'Movies\\imgopt2_%03d.png','-c:v', 'libx264', '-pix_fmt', 'yuv420p', 'opt.mp4'])

def stokeslet_vec(x, e):
    # Spagnolie & Lauga 2012 notation
    x=np.array(x)
    e=np.array(e)
    rinvsq = 1.0/np.dot(x, x)
    rinv = 1.0/np.linalg.norm(x)
    vec = rinv*(e + np.dot(x,e)*x*rinvsq)
    return vec

def Lin_ODE(t,y,args):
    v1,v2,x1,x2=args
    y0=y[0]
    y1=y[1]
    y_velocity1=6*np.pi*stokeslet_vec([y0-x1,y1],[v1,0])
    #contribution from first active particle
    y_velocity2=6*np.pi*stokeslet_vec([y0,y1-x2],[0,v2])
    #contribution from second active particle
    return 100*(y_velocity1+y_velocity2)/(8*np.pi)

def lin_solver(v1vals,v2vals,x1vals,x2vals,y0,tvals):
    dt=tvals[1]-tvals[0]
    trajectory=np.empty((tvals.size,2)) #trajectory of particle
    Solver=odes.ode(Lin_ODE).set_integrator('DOP853',max_step=dt)
    Solver.set_initial_value(y0,0)

    print('Solving ODE...')
    for i in range(len(tvals)):
        Solver.set_f_params([v1vals[i],v2vals[i],x1vals[i],x2vals[i]])
        trajectory[i,:]=np.array(Solver.integrate(Solver.t+dt))   
    return trajectory

fig2 = pl.figure()
trajectory=lin_solver(v1.value,v2.value,x1.value,x2.value,[-30,-30],g.time)
pl.plot(trajectory[:,0],trajectory[:,1], label = 'Actual Trajectory')
pl.plot(y1.value,y2.value,label='Predicted Trajectory')
pl.legend()
pl.show()
