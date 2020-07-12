from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as pl
import subprocess
import time 
import scipy.integrate as odes
import subprocess
from scipy.spatial.transform import Rotation as Rrr

g=GEKKO(remote=True)

nt =100 #number of timeplots
g.time=np.linspace(0,100,nt)

#STATUS = 1 implies that the variable is being optimized by the solver
#DCOST is the amount which is added to the cost function when the variable is modified -> this prevents blowup

# These are the coordinates of the passive particle
y1=g.CV(value=0)
y1.LOWER=-100
y1.UPPER=100
y2=g.CV(value=0)
y2.LOWER=-100
y2.UPPER=100

x11=g.Var(value=1)
x12=g.Var(value=0)
x21=g.Var(value=0)
x22=g.Var(value=1)
x11.LOWER=-1
x11.UPPER=1
x12.LOWER=-1
x12.UPPER=1
x21.LOWER=-1
x21.UPPER=1
x22.LOWER=-1
x22.UPPER=1
v11=g.MV();v11.STATUS=1
v12=g.MV();v12.STATUS=1
v21=g.MV();v21.STATUS=1
v22=g.MV();v22.STATUS=1
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
g.Minimize(1e3*((x11-y1)**2+(x21-y1)**2+(x12-y2)**2+(x22-y2)**2)**(-1))

def stokeslet(y1,y2,x11,x12,x21,x22,v11,v12,v21,v22):
    k=3/4
    r11=y1-x11
    r12=y2-x12
    r21=y1-x21
    r22=y2-x22
    rinvsq1=1/(r11**2+r12**2)
    rinv1=rinvsq1**0.5
    rinvsq2=1/(r21**2+r22**2)
    rinv2=rinvsq2**0.5 
    rv1=r11*v11+r12*v12 
    rv2=r21*v21+r22*v22
    ydot1=rinv1*(v11+rv1*r11*rinvsq1)+rinv2*(v21+rv2*r21*rinvsq2) 
    ydot2=rinv1*(v12+rv1*r12*rinvsq1)+rinv2*(v22+rv2*r22*rinvsq2)
    return k*np.array([ydot1,ydot2])

# Dynamical constraints

g.Equation(y1.dt()==stokeslet(y1,y2,x11,x12,x21,x22,v11,v12,v21,v22)[0])
g.Equation(y2.dt()==stokeslet(y1,y2,x11,x12,x21,x22,v11,v12,v21,v22)[1])

# Cost function
J=g.Var(value=0)

g.Equation(J.dt()==v11**2+v12**2+v21**2+v22**2)

final = g.Param(np.zeros(nt)); final[-1]=1
g.Minimize(J*final)

g.Minimize(final*1e5*(y1-3)**2)
g.Minimize(final*1e5*(y2-3)**2)

g.options.IMODE = 6  # optimal control
g.options.NODES = 8  # collocation nodes
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
ax2.plot(g.time,v11.value,label='v11')
ax2.plot(g.time,v12.value,label='v12')
ax2.plot(g.time,v21.value,label='v21')
ax2.plot(g.time,v22.value,label='v22')
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
    ax.scatter(x11.value[counter],x12.value[counter],c='r')
    ax.scatter(x21.value[counter],x22.value[counter],c='r')

    pl.savefig('Movies\\imgopt3_'+str("%03d"%(counter))+'.png')
    pl.close()
    print(counter)
    counter=counter+1
subprocess.call(['ffmpeg','-y', '-i', 'Movies\\imgopt3_%03d.png','-c:v', 'libx264', '-pix_fmt', 'yuv420p', 'opt.mp4'])

def stokeslet_vec(x, e):
    # Spagnolie & Lauga 2012 notation
    x=np.array(x)
    e=np.array(e)
    rinvsq = 1.0/np.dot(x, x)
    rinv = 1.0/np.linalg.norm(x)
    vec = rinv*(e + np.dot(x,e)*x*rinvsq)
    return vec

def Lin_ODE(t,y,args):
    v11,v12,v21,v22,x11,x12,x21,x22=args
    y1=y[0]
    y2=y[1]
    y_velocity1=(3/4)*stokeslet_vec([y1-x11,y2-x12],[v11,v12])
    #contribution from first active particle
    y_velocity2=(3/4)*stokeslet_vec([y1-x21,y2-x22],[v21,v22])
    #contribution from second active particle
    return y_velocity1+y_velocity2

def lin_solver(v11vals,v12vals,v21vals,v22vals,x11vals,x12vals,x21vals,x22vals,y0,tvals):
    dt=tvals[1]-tvals[0]
    trajectory=np.empty((tvals.size,2)) #trajectory of particle
    Solver=odes.ode(Lin_ODE).set_integrator('DOP853',max_step=dt)
    Solver.set_initial_value(y0,0)

    print('Solving ODE...')
    for i in range(len(tvals)):
        Solver.set_f_params([v11vals[i],v12vals[i],v21vals[i],v22vals[i],x11vals[i],x12vals[i],x21vals[i],x22vals[i]])
        trajectory[i,:]=np.array(Solver.integrate(Solver.t+dt))   
    return trajectory

fig2 = pl.figure()
trajectory=lin_solver(v11.value,v12.value,v21.value,v22.value,x11.value,x12.value,x21.value,x22.value,[0,0],g.time)
pl.plot(trajectory[:,0],trajectory[:,1], label = 'Actual Trajectory')
pl.plot(y1.value,y2.value,label='Predicted Trajectory')
pl.legend()
pl.show()
