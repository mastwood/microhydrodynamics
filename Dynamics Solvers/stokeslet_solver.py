# Optimized version of main.py to generate stokes flow trajectories
"""
Created on 2020-05-25
@author: Michael Astwood, Henry Shum
"""

import time 
import numpy as np 
import matplotlib.pyplot as pl 
import scipy.integrate as odes
import quaternions
import subprocess
from scipy.spatial.transform import Rotation as Rrr
#import StokesSingularities as ss 

# grid setup
x=np.linspace(-15,15,31)
y=x
z=x
positions=np.mgrid[0:30.0:31j,
                      -5:5:31j,
                      -15:15:31j]
xx, yy, zz = positions

#begin timer
tic=time.perf_counter()

def stokeslet_vec(x, e):
    # Spagnolie & Lauga 2012 notation
    rinvsq = 1.0/np.dot(x, x)
    rinv = 1.0/np.linalg.norm(x)
    
    vec = (3/4)*rinv*(e + np.dot(x,e)*x*rinvsq)
    
    return vec
def stresslet_vec(x, d, e):
    # Spagnolie & Lauga 2012 notation
    return (-6.0*(1.0/np.dot(x, x))**2*(1.0/np.linalg.norm(x))*np.dot(d,x)*np.dot(e,x)*x)


tfinal=1000
dt=0.01
TT=np.arange(0,tfinal,dt) #main timeseries variable
trajectory1=np.empty((TT.size,3)) #trajectory of particle
trajectory2=np.empty((TT.size,3)) #trajectory of particle
trajectory3=np.empty((TT.size,3)) #trajectory of particle
trajectory4=np.empty((TT.size,3)) #trajectory of particle

singularity_pos = lambda t: np.array([0.0,5*np.cos(2*np.pi*t),5*np.sin(2*np.pi*t)],dtype=object)

def singularity_pos20(t):
    t = t-np.floor(t)
    if t <= 0.25:
        return np.array([0.0,np.cos(2*np.pi*t),np.sin(2*np.pi*t)])
    elif 0.25 < t and t <= 0.5:
        return np.array([0.0,np.cos(2*np.pi*0.25),np.sin(2*np.pi*0.25)])
    elif 0.5 < t and t <= 0.75:
        return np.array([0.0,np.cos(2*np.pi*(0.75-t)),np.sin(2*np.pi*(0.75-t))])
    else:
        return np.array([0.0,np.cos(2*np.pi*0),np.sin(2*np.pi*0)])

singularity_position=np.array([singularity_pos(t) for t in TT])
singularity_position2=np.array([singularity_pos20(t) for t in TT])
singularity_position3=np.array([singularity_pos20(t+0.25) for t in TT])
singularity_velocity = lambda t: np.array([0.0,-5*2*np.pi*np.sin(2*np.pi*t),5*2*np.pi*np.cos(2*np.pi*t)])
def singularity_velocity20(t):
    t = t-np.floor(t)
    if t <= 0.25:
        return 2*np.pi*np.array([0.0,np.sin(2*np.pi*t),-np.cos(2*np.pi*t)])
    elif 0.25 < t and t <= 0.5:
        return np.array([0,0,0])
    elif 0.5 < t and t <= 0.75:
        return -2*np.pi*np.array([0.0,np.sin(2*np.pi*(t)),-np.cos(2*np.pi*(t))])
    else:
        return np.array([0,0,0])
singularity_velocity2=np.vectorize(singularity_velocity20)

stokeslet_vector = lambda x,t: stokeslet_vec(x,singularity_velocity(t))
stokeslet_vector2 = lambda x,t: stokeslet_vec(x,singularity_velocity2(t))
#Rot0 = Rrr.from_quat([np.sin(np.pi/4), np.cos(np.pi/4),0,0]).as_matrix()
Rot0 = Rrr.from_quat([-1,0,0,0]).as_matrix()

stokeslet_vectorRrr = lambda x,t: stokeslet_vec(np.matmul(Rot0,x),np.matmul(Rot0,singularity_velocity2(t)))
rotator = lambda x: np.matmul(Rot0,x)
singularity_pos_rot = np.array([rotator(singularity_pos20(t-0.25)) for t in TT])

#Solve ode
def velocity(t,y):
    return stokeslet_vector(y-singularity_pos(t),t)
def velocity2(t,y):
    return stokeslet_vector2(y-singularity_pos20(t),t)+stokeslet_vector2(y-singularity_pos(t+0.25),t+0.25)

trajectories=[]

# r1=odes.ode(velocity2).set_integrator("lsoda")
# r1.set_initial_value(np.array([0,0,5]),0.0)
# for i in range(len(list(TT))):
#     x=np.array(r1.integrate(r1.t+dt))
#     trajectory1[i,:] = x
# trajectories.append(trajectory1)

# r2=odes.ode(velocity2).set_integrator("lsoda")
# r2.set_initial_value(np.array([0,0,0]),0.0)
# for i in range(len(list(TT))):
#     x=np.array(r2.integrate(r2.t+dt))
#     trajectory2[i,:] = x

# trajectories.append(trajectory2)

# r3=odes.ode(velocity2).set_integrator("lsoda")
# r3.set_initial_value(np.array([0,5,0]),0.0)
# for i in range(len(list(TT))):
#     x=np.array(r3.integrate(r3.t+dt))
#     trajectory3[i,:] = x

# trajectories.append(trajectory3)

r4=odes.ode(velocity2).set_integrator("rk45")
r4.set_initial_value(np.array([0,0,20]),0.0)
for i in range(len(list(TT))):
    print(i)
    x=np.array(r4.integrate(r4.t+dt))
    trajectory4[i,:] = x

trajectories.append(trajectory4)

# #print(trajectory)

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
from mpl_toolkits.mplot3d import Axes3D

fig = pl.figure()
ax = fig.add_subplot(111)
ax.grid()
ax.plot(trajectories[0][:,1],trajectories[0][:,2],'k')

#x_params = np.polyfit(TT, trajectories[0][:,1], 2)
#y_params = np.polyfit(TT, trajectories[0][:,2], 2)
#xpxp = np.poly1d(x_params)
#ypyp = np.poly1d(y_params)
#ax.plot(xpxp(TT),ypyp(TT),'k--')
# ax.plot(trajectories[1][:,0],trajectories[1][:,1],trajectories[1][:,2],lw=0.5)
# ax.plot(trajectories[2][:,0],trajectories[2][:,1],trajectories[2][:,2],lw=0.5)
# ax.plot(trajectories[3][:,0],trajectories[3][:,1],trajectories[3][:,2],lw=0.5)
ax.plot(singularity_position2[:,1],singularity_position2[:,2],'--k',lw=0.75)
ax.plot(singularity_position3[:,1],singularity_position3[:,2],'--k',lw=0.75)
# ax.plot(singularity_pos_rot[:,0],singularity_pos_rot[:,1],singularity_pos_rot[:,2],lw=0.5)
ax.set_xlabel('y')
ax.set_ylabel('z')
ax.set_aspect(1)
pl.show()

# #print(singularity_position[1])
# counter=0
# while counter < len(TT):
#     fig = pl.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.set_xlim3d(-5,5)
#     ax.set_ylim3d(-5,5)
#     ax.set_zlim3d(-5,5)
    
#     ax.plot(trajectories[0][0:counter,0],trajectories[0][0:counter,1],trajectories[0][0:counter,2],lw=0.5)
#     ax.plot(trajectories[1][0:counter,0],trajectories[1][0:counter,1],trajectories[1][0:counter,2],lw=0.5)
#     ax.plot(trajectories[2][0:counter,0],trajectories[2][0:counter,1],trajectories[2][0:counter,2],lw=0.5)
#     ax.plot(trajectories[3][0:counter,0],trajectories[3][0:counter,1],trajectories[3][0:counter,2],lw=0.5)

#     ax.scatter(trajectories[0][counter-1:counter,0],trajectories[0][counter-1:counter,1],trajectories[0][counter-1:counter,2])
#     ax.scatter(trajectories[1][counter-1:counter,0],trajectories[1][counter-1:counter,1],trajectories[1][counter-1:counter,2])
#     ax.scatter(trajectories[2][counter-1:counter,0],trajectories[2][counter-1:counter,1],trajectories[2][counter-1:counter,2])
#     ax.scatter(trajectories[3][counter-1:counter,0],trajectories[3][counter-1:counter,1],trajectories[3][counter-1:counter,2])

#     ax.scatter(singularity_position2[counter-1:counter,0],singularity_position2[counter-1:counter,1],singularity_position2[counter-1:counter,2],lw=0.5)
#     ax.scatter(singularity_pos_rot[counter-1:counter,0],singularity_pos_rot[counter-1:counter,1],singularity_pos_rot[counter-1:counter,2],lw=0.5)
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')
#     set_axes_equal(ax)

#     pl.savefig('Movies\\img5_'+str("%03d"%(counter/25))+'.png')
#     pl.close()
#     print(counter)
#     counter=counter+25
# subprocess.call(['ffmpeg','-y', '-i', 'Movies\\img5_%03d.png','-c:v', 'libx264', '-pix_fmt', 'yuv420p', 'out4.mp4'])

# #pl.show()
# if 0:
#     quivskip=2
#     for T in TT:
#         ux = np.zeros_like(xx)
#         uy = np.zeros_like(xx)
#         uz = np.zeros_like(xx)
#         for i in range(xx.size):
#             X = np.array([xx.flat[i],yy.flat[i],zz.flat[i]])
#             vel = stokeslet_vec(X-singularity_pos(T), -1.0*singularity_velocity(T))
#             ux.flat[i] = vel[0]
#             uy.flat[i] = vel[1]
#             uz.flat[i] = vel[2]
        
#         #reshape the datapoints
#         pltxx = np.reshape(xx[:,0,:],(xx.shape[0],-1))
#         pltyy = np.reshape(zz[:,0,:],(xx.shape[0],-1))
#         pltux = np.reshape(ux[:,0,:],(xx.shape[0],-1))
#         pltuy = np.reshape(uz[:,0,:],(xx.shape[0],-1))
#         spd = (ux**2 + uy**2 + uz**2)**0.5 #speed of fluid
#         pltspd = np.reshape(spd[:,0,:],(xx.shape[0],-1))
#         quivi, quivj = np.mgrid[0:pltxx.shape[0]:quivskip, 0:pltxx.shape[1]:quivskip]

#         #fig setup
#         fig = pl.figure(figsize=(12,6))
#         ax = fig.gca()
#         ax.set_aspect('equal')
#         #plot the speed
#         splt = ax.contourf(pltxx, pltyy, np.log(pltspd),alpha=0.4)
#         fig.colorbar(splt) 

#         #plot the vector field
#         ax.quiver(pltxx[quivi,quivj], pltyy[quivi,quivj], 
#                 pltux[quivi,quivj]/pltspd[quivi,quivj], 
#                 pltuy[quivi,quivj]/pltspd[quivi,quivj])         

#         #plot the streamline
#         ax.streamplot(pltxx.transpose(), pltyy.transpose(), 
#                 pltux.transpose(), pltuy.transpose(), color='r')

#         #plot the test particle
#         pl.plot(trajectory[0:TT.tolist().index(T),0],trajectory[0:TT.tolist().index(T),2])

#         pl.savefig('stokeslet_movie\\img3_'+str("%02d"%(TT.tolist().index(T)))+'.png')
#         pl.close()
#         print(TT.tolist().index(T))
#     subprocess.call(['ffmpeg','-y', '-i', 'stokeslet_movie\\img3_%02d.png','-c:v', 'libx264', '-pix_fmt', 'yuv420p', 'out2.mp4'])

# toc=time.perf_counter()
# print("Completed in " + str(toc-tic) + " seconds.")
#---------








