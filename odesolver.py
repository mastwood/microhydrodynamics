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
    
    vec = rinv*(e + np.dot(x,e)*x*rinvsq)
    
    return vec
def stresslet_vec(x, d, e):
    # Spagnolie & Lauga 2012 notation
    return (-6.0*(1.0/np.dot(x, x))**2*(1.0/np.linalg.norm(x))*np.dot(d,x)*np.dot(e,x)*x)


tfinal=20
dt=0.001
TT=np.arange(0,tfinal,dt) #main timeseries variable
trajectory=np.empty((TT.size,3)) #trajectory of particle

singularity_pos = lambda t: np.array([0.0,5*np.cos(2*np.pi*t),5*np.sin(2*np.pi*t)])

def singularity_pos20(t):
    t = t-np.floor(t)
    if t <= 0.25:
        return np.array([0.0,np.cos(4*np.pi*t),np.sin(4*np.pi*t)])
    elif 0.25 < t and t <= 0.5:
        return np.array([0.0,np.cos(4*np.pi*0.25),np.sin(4*np.pi*0.25)])
    elif 0.5 < t and t <= 0.75:
        return np.array([0.0,np.cos(4*np.pi*(0.25-t)),np.sin(4*np.pi*(0.25-t))])
    else:
        return np.array([0.0,np.cos(4*np.pi*0),np.sin(4*np.pi*0)])
singularity_pos2=np.vectorize(singularity_pos20)

singularity_position=singularity_pos(TT)
singularity_position2=np.array([singularity_pos20(t) for t in TT])

singularity_velocity = lambda t: np.array([0.0,-5*2*np.pi/100*np.sin(2*np.pi*t/100),5*2*np.pi/100*np.cos(2*np.pi*t/100)])
def singularity_velocity20(t):
    t = t-np.floor(t)
    if t <= 0.25:
        return 4*np.pi*np.array([0.0,np.sin(4*np.pi*t),-np.cos(4*np.pi*t)])
    elif 0.25 < t and t <= 0.5:
        return 4*np.pi*np.array([0.0,np.sin(4*np.pi*0.25),-np.cos(4*np.pi*0.25)])
    elif 0.5 < t and t <= 0.75:
        return 4*np.pi*np.array([0.0,np.sin(4*np.pi*(0.5-t)),-np.cos(4*np.pi*(0.5-t))])
    else:
        return 4*np.pi*np.array([0.0,np.sin(4*np.pi*0),-np.cos(4*np.pi*0)])
singularity_velocity2=np.vectorize(singularity_velocity20)

stokeslet_vector = lambda x,t: stokeslet_vec(x,singularity_velocity(t))
stokeslet_vector2 = lambda x,t: stokeslet_vec(x,singularity_velocity2(t))

#Solve ode
def velocity(t,y):
    return stokeslet_vector(y-singularity_pos(t),t)
def velocity2(t,y):
    return stokeslet_vector2(y-singularity_pos2(t),t)-stokeslet_vector2(y+singularity_pos2(t+0.25),t)

r=odes.ode(velocity2).set_integrator("lsoda")
r.set_initial_value(np.array([5.0,0.0,0.0]),0.0)
for i in range(len(list(TT))):
    x=np.array(r.integrate(r.t+dt))
    trajectory[i,:] = x
#print(trajectory)

from mpl_toolkits.mplot3d import Axes3D
print(trajectory)
fig = pl.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(trajectory[:,0],trajectory[:,1],trajectory[:,2])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
# pl.plot(trajectory[:,1],trajectory[:,2])
ax.plot(np.zeros(TT.shape),singularity_position2[:,1],singularity_position2[:,2])
pl.show()

#print(singularity_position[1])
counter=0
# while counter < len(TT):
#     fig = pl.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     ax.plot(trajectory[0:counter,0],trajectory[0:counter,1],trajectory[0:counter,2])
#     ax.plot(np.zeros(counter),singularity_position[1][0:counter],singularity_position[2][0:counter])

#     u,v,w=stokeslet_vector(trajectory[counter,:],TT[counter])*10
#     ax.quiver(trajectory[counter,0],trajectory[counter,1],trajectory[counter,2],u,v,w,color='r')

#     ax.quiver(0,singularity_position[1][counter],singularity_position[2][counter],
#         singularity_velocity(TT[counter])[0],singularity_velocity(TT[counter])[1],singularity_velocity(TT[counter])[2],color='r')

#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')
    
#     ax.set_xlim(0,5)
#     ax.set_ylim(-5,5)
#     ax.set_zlim(-5,5)

#     pl.savefig('stokeslet_movie\\img4_'+str("%02d"%(counter/5))+'.png')
#     pl.close()
#     print(counter)
#     counter=counter+5
# subprocess.call(['ffmpeg','-y', '-i', 'stokeslet_movie\\img4_%02d.png','-c:v', 'libx264', '-pix_fmt', 'yuv420p', 'out3.mp4'])

#pl.show()
if 0:
    quivskip=2
    for T in TT:
        ux = np.zeros_like(xx)
        uy = np.zeros_like(xx)
        uz = np.zeros_like(xx)
        for i in range(xx.size):
            X = np.array([xx.flat[i],yy.flat[i],zz.flat[i]])
            vel = stokeslet_vec(X-singularity_pos(T), -1.0*singularity_velocity(T))
            ux.flat[i] = vel[0]
            uy.flat[i] = vel[1]
            uz.flat[i] = vel[2]
        
        #reshape the datapoints
        pltxx = np.reshape(xx[:,0,:],(xx.shape[0],-1))
        pltyy = np.reshape(zz[:,0,:],(xx.shape[0],-1))
        pltux = np.reshape(ux[:,0,:],(xx.shape[0],-1))
        pltuy = np.reshape(uz[:,0,:],(xx.shape[0],-1))
        spd = (ux**2 + uy**2 + uz**2)**0.5 #speed of fluid
        pltspd = np.reshape(spd[:,0,:],(xx.shape[0],-1))
        quivi, quivj = np.mgrid[0:pltxx.shape[0]:quivskip, 0:pltxx.shape[1]:quivskip]

        #fig setup
        fig = pl.figure(figsize=(12,6))
        ax = fig.gca()
        ax.set_aspect('equal')
        #plot the speed
        splt = ax.contourf(pltxx, pltyy, np.log(pltspd),alpha=0.4)
        fig.colorbar(splt) 

        #plot the vector field
        ax.quiver(pltxx[quivi,quivj], pltyy[quivi,quivj], 
                pltux[quivi,quivj]/pltspd[quivi,quivj], 
                pltuy[quivi,quivj]/pltspd[quivi,quivj])         

        #plot the streamline
        ax.streamplot(pltxx.transpose(), pltyy.transpose(), 
                pltux.transpose(), pltuy.transpose(), color='r')

        #plot the test particle
        pl.plot(trajectory[0:TT.tolist().index(T),0],trajectory[0:TT.tolist().index(T),2])

        pl.savefig('stokeslet_movie\\img3_'+str("%02d"%(TT.tolist().index(T)))+'.png')
        pl.close()
        print(TT.tolist().index(T))
    subprocess.call(['ffmpeg','-y', '-i', 'stokeslet_movie\\img3_%02d.png','-c:v', 'libx264', '-pix_fmt', 'yuv420p', 'out2.mp4'])

toc=time.perf_counter()
print("Completed in " + str(toc-tic) + " seconds.")
#---------








