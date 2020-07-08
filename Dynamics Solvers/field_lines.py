# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 18:13:49 2019

@author: Henry Shum
"""
import time
import numpy as np
import StokesSingularities as ss
import matplotlib.pyplot as plt
import quaternions
from matplotlib.animation import FuncAnimation
import subprocess

x0 = np.array([0,2,5])  # Singularity position
phi0 = np.pi*0.5 # Singularity orientation

#xx, yy, zz = np.mgrid[-4.75:4.75:10j,
#                      -4.75:4.75:10j,
#                      0:0.5:1j]
xx, yy, zz = np.mgrid[-15.0:15.0:31j,
                      0:0:31j,
                      -15.0:15:31j]



ux = np.zeros_like(xx)
uy = np.zeros_like(xx)
uz = np.zeros_like(xx)
uxim = np.zeros_like(xx)
uyim = np.zeros_like(xx)
uzim = np.zeros_like(xx)


q0 = np.hstack((np.cos(phi0/2),np.sin(phi0/2)*np.array([0,-1,0])))
x0im = np.array([x0[0], x0[1], -x0[2]])

def velfield(x, x0, q, S):
    
    x0im = x0.copy()
    x0im[2] = -x0[2]
    dx = x - x0
    dxim = x - x0im
    
    R = quaternions.quat2mat(q)
    Slab = np.matmul(R, np.matmul(S, R.transpose()))
    #print(Slab)
    vel = np.zeros(3)
    velim = np.zeros(3)
    
    for i in range(3):
        d = np.zeros(3)
        d[i] = 1.0
        for j in range(3):
            e = np.zeros(3)
            e[j] = 1.0
            
            fun = lambda r: ss.stressletim_vec(r, d, e, x0[2])
            
            vij = fun(dxim)
            
            velim += Slab[i,j]*vij
            vel += Slab[i,j]*ss.stresslet_vec(dx, d, e)
    
    return vel, velim   



if 1:  # Animation
    tic=time.perf_counter()
    
    #Compute trajectory of particle under moving stokeslet flow
    dt=5
    times = np.arange(0,100,dt)
    Y0 = np.array([-5,10,0])
    Yold=Y0
    YX=np.empty(len(list(times)))
    YY=np.empty(len(list(times)))
    YZ=np.empty(len(list(times)))
    counter=0
    for t in times:
        x0=np.array([np.sin(t/10), 0.0, np.cos(t/10)])
        
        Fvec = np.array([2.0, 0.0, 0.0])
        print(counter)
        counter=counter+1
        for i in range(xx.size):
            X = np.array([xx.flat[i],yy.flat[i],zz.flat[i]])
            vel = ss.stokeslet_vec(X-x0, Fvec)
            #display(vel - ss.stokeslet_vec(X-x0, Fvec))
            ux.flat[i] = vel[0]
            uy.flat[i] = vel[1]
            uz.flat[i] = vel[2]
        

        uxshow = ux + uxim
        uyshow = uy + uyim
        uzshow = uz + uzim
        spd = (uxshow**2 + uyshow**2 + uzshow**2)**0.5

        secdim = 1
        seci = 0
        quivskip = 2

        if secdim==1:
            pltxx = np.reshape(xx[:,0,:],(xx.shape[0],-1))
            pltyy = np.reshape(zz[:,0,:],(xx.shape[0],-1))
            pltspd = np.reshape(spd[:,seci,:],(xx.shape[0],-1))
            pltux = np.reshape(uxshow[:,seci,:],(xx.shape[0],-1))
            pltuy = np.reshape(uzshow[:,seci,:],(xx.shape[0],-1))
        elif secdim==2:
            pltxx = xx[:,:,0]
            pltyy = yy[:,:,0]
            pltspd = spd[:,:,seci]
            pltux = uxshow[:,:,seci]
            pltuy = uyshow[:,:,seci]       
            
        uxx, uxy = np.gradient(pltux,pltxx[:,0],pltyy[0,:])
        uyx, uyy = np.gradient(pltuy,pltxx[:,0],pltyy[0,:])
        vort = uyx - uxy
        fig = plt.figure(figsize=(12,6))
        ax = fig.gca()
        ax.set_aspect('equal')

        splt = ax.contourf(pltxx, pltyy, vort)
        #splt = ax.contourf(pltxx, pltyy, np.log(pltspd))
        #ax.quiver(xx, zz, ux/spd, uz/spd)
        #ax.quiver(pltxx, pltyy, pltux, pltuy)
        quivi, quivj = np.mgrid[0:pltxx.shape[0]:quivskip, 0:pltxx.shape[1]:quivskip]

        ax.quiver(pltxx[quivi,quivj], pltyy[quivi,quivj], 
                pltux[quivi,quivj]/pltspd[quivi,quivj], 
                pltuy[quivi,quivj]/pltspd[quivi,quivj])

        ax.streamplot(pltxx.transpose(), pltyy.transpose(), 
                    pltux.transpose(), pltuy.transpose(), color='r')
        fig.colorbar(splt)


        plt.savefig('stokeslet_movie\\img2_'+str("%02d"%(times.tolist().index(t)))+'.png')
        plt.close()
    subprocess.call(['ffmpeg','-y', '-i', 'stokeslet_movie\\img2_%02d.png','-c:v', 'libx264', '-pix_fmt', 'yuv420p', 'out3.mp4'])
    toc=time.perf_counter()
    print(f'Completed in {toc-tic:0.4f} seconds')

elif 0:  # Stresslet
    
    Fvec = np.array([0.0, 0.0, -1.0])
    evec = np.array([0.0, 0.0, 1.0])
    R = quaternions.quat2mat(q0)
    Flab = np.matmul(R,Fvec)
    elab = np.matmul(R,evec)
    Smat = np.outer(Fvec,evec)
    
    for i in range(xx.size):
        X = np.array([xx.flat[i],yy.flat[i],zz.flat[i]])
        #mat = ss.stresslet_tens(x0, X)
        #vel = np.tensordot(Smat, mat)
        vel = ss.stresslet_vec(X-x0, Flab, elab)
        #display(vel - ss.stresslet_vec(X-x0, Fvec, evec))
        ux.flat[i] = vel[0]
        uy.flat[i] = vel[1]
        uz.flat[i] = vel[2]
            

elif 0:  # Stresslet with image
    
    Fvec = np.array([1.0, 0.0, 0.0])
    evec = np.array([1.0, 0.0, 0.0])
    R = quaternions.quat2mat(q0)
    Flab = np.matmul(R,Fvec)
    elab = np.matmul(R,evec)
    Smat = np.outer(Fvec,evec)
    
    for i in range(xx.size):
        X = np.array([xx.flat[i],yy.flat[i],zz.flat[i]])
        #mat = ss.stresslet_tens(x0, X)
        #vel = np.tensordot(Smat, mat)
        vel = ss.stresslet_vec(X-x0, Flab, elab)
        velim = ss.stressletim_vec(X-x0im, Flab, elab, x0[2])
        #display(vel - ss.stresslet_vec(X-x0, Fvec, evec))
        ux.flat[i] = vel[0]
        uy.flat[i] = vel[1]
        uz.flat[i] = vel[2]
        uxim.flat[i] = velim[0]
        uyim.flat[i] = velim[1]
        uzim.flat[i] = velim[2]
        
elif 0:  # Stresslet image only
    
    Fvec = np.array([1.0, 0.0, 0.0])
    evec = np.array([1.0, 0.0, 0.0])
    R = quaternions.quat2mat(q0)
    Flab = np.matmul(R,Fvec)
    elab = np.matmul(R,evec)
    Smat = np.outer(Fvec,evec)
    
    for i in range(xx.size):
        X = np.array([xx.flat[i],yy.flat[i],zz.flat[i]])
        #mat = ss.stresslet_tens(x0, X)
        #vel = np.tensordot(Smat, mat)
        #vel = ss.stresslet_vec(X-x0, Flab, elab)
        velim = ss.stressletim_vec(X-x0im, Flab, elab, x0[2])
        #display(vel - ss.stresslet_vec(X-x0, Fvec, evec))
        ux.flat[i] = velim[0]
        uy.flat[i] = velim[1]
        uz.flat[i] = velim[2]
        
elif 0:  # General Stresslet strength
    
    Fvec = np.array([-1.0, 0.0, 0.0])
    evec = np.array([1.0, 0.0, 0.0])
    Smat = np.outer(Fvec,evec)
    Smat = 0.5*(Smat + Smat.transpose()) - np.trace(Smat)/3*np.eye(3)
    
    S=np.zeros((3,3))
    S[0,0] = 1.0
    S[1,1] = -2.0
    S[2,2] = 1.00
    
    #S[0,2] = 0.5
    #S[2,0] = S[0,2]

#    S[0,0] = -(96*np.pi + 32)/(9*np.pi + 6)
#    #S[0,2] = 
#    S[1,1] = (48*np.pi + 16)/(9*np.pi + 6)
#    S[2,2] = (48*np.pi + 16)/(9*np.pi + 6)
    Smat = -0.5*S/(8*np.pi)
    
    for i in range(xx.size):
        X = np.array([xx.flat[i],yy.flat[i],zz.flat[i]])
        print(i)
        
        #mat = ss.stresslet_tens(x0, X)
        #vel = np.tensordot(Smat, mat)
        #vel = ss.stresslet_vec(X-x0, Flab, elab)
        vel, velim = velfield(X, x0, q0, Smat)
        #display(vel - ss.stresslet_vec(X-x0, Fvec, evec))

            
        ux.flat[i] = vel[0]
        uy.flat[i] = vel[1]
        uz.flat[i] = vel[2]
        uxim.flat[i] = velim[0]
        uyim.flat[i] = velim[1]
        uzim.flat[i] = velim[2]
            


# %% Plot

showmode = 2  # Original, image, or both

if showmode == 0: # Only original singularity
    uxshow = ux
    uyshow = uy
    uzshow = uz
elif showmode == 1:  # Only image singularities
    uxshow = uxim
    uyshow = uyim
    uzshow = uzim
else:
    uxshow = ux + uxim
    uyshow = uy + uyim
    uzshow = uz + uzim

spd = (uxshow**2 + uyshow**2 + uzshow**2)**0.5

secdim = 1
seci = 0
quivskip = 2

if secdim==1:
    pltxx = np.reshape(xx[:,0,:],(xx.shape[0],-1))
    pltyy = np.reshape(zz[:,0,:],(xx.shape[0],-1))
    pltspd = np.reshape(spd[:,seci,:],(xx.shape[0],-1))
    pltux = np.reshape(uxshow[:,seci,:],(xx.shape[0],-1))
    pltuy = np.reshape(uzshow[:,seci,:],(xx.shape[0],-1))
elif secdim==2:
    pltxx = xx[:,:,0]
    pltyy = yy[:,:,0]
    pltspd = spd[:,:,seci]
    pltux = uxshow[:,:,seci]
    pltuy = uyshow[:,:,seci]
    
    
uxx, uxy = np.gradient(pltux,pltxx[:,0],pltyy[0,:])
uyx, uyy = np.gradient(pltuy,pltxx[:,0],pltyy[0,:])
vort = uyx - uxy


fig = plt.figure(figsize=(12,6))
ax = fig.gca()
ax.set_aspect('equal')

splt = ax.contourf(pltxx, pltyy, vort)
#splt = ax.contourf(pltxx, pltyy, np.log(pltspd))
#ax.quiver(xx, zz, ux/spd, uz/spd)
#ax.quiver(pltxx, pltyy, pltux, pltuy)
quivi, quivj = np.mgrid[0:pltxx.shape[0]:quivskip, 0:pltxx.shape[1]:quivskip]

ax.quiver(pltxx[quivi,quivj], pltyy[quivi,quivj], 
          pltux[quivi,quivj]/pltspd[quivi,quivj], 
          pltuy[quivi,quivj]/pltspd[quivi,quivj])

ax.streamplot(pltxx.transpose(), pltyy.transpose(), 
              pltux.transpose(), pltuy.transpose(), color='r')
fig.colorbar(splt)

plt.show()

