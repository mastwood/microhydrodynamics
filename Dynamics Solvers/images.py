import numpy as np
from numpy.core.numeric import empty_like
import scipy.integrate as si
import matplotlib.pyplot as pl 
from matplotlib import animation 
#from numba import jit, njit, vectorize
from mpl_toolkits.axes_grid1 import make_axes_locatable
a=0.1


def stokeslet_vec(x, e):
    # Spagnolie & Lauga 2012 notation
    rinvsq = 1.0/np.dot(x, x)
    rinv = 1.0/np.linalg.norm(x)
    
    vec = rinv*(e + np.dot(x,e)*x*rinvsq)
    
    return vec


def stokesletdipole_vec(x, d, e):
    # Spagnolie & Lauga 2012 notation
    rinvsq = 1.0/np.dot(x, x)
    rinv = 1.0/np.linalg.norm(x)
    
    vec = rinvsq*(rinv*(np.dot(d,x)*e - np.dot(e,x)*d - np.dot(d,e)*x) +
                  3.0*np.dot(e,x)*np.dot(d,x)*x*rinv*rinvsq)
    
    return vec


def sourcedoublet_vec(x, e):
    # Spagnolie & Lauga 2012 notation
    rinvsq = 1.0/np.dot(x, x)
    rinv = 1.0/np.linalg.norm(x)

    vec = rinv*rinvsq*(-e + 3.0*np.dot(x,e)*x*rinvsq)
    
    return vec


def blakelet_vec(y,x,e):
    # Spagnolie & Lauga 2012 notation
    # No slip plane boundary at z=0. h is z co-ordinate of stokeslet
    # x = position of active particle
    # y = position of passive particle
    # e = direction of motion of active particle
    hvec=np.zeros_like(x)
    hvec[-1]=x[-1]
    h = x[-1]
    zhat=np.zeros_like(x)
    zhat[-1]=1
    r = y-x
    rprime = y-(x-2*hvec)
    ephi = np.zeros_like(x)
    ephi[0]=1

    vec = stokeslet_vec(r,e) + (-1*stokeslet_vec(rprime,ephi)+2*h*stokesletdipole_vec(rprime,ephi,zhat)-2*h**2*sourcedoublet_vec(rprime,ephi))*e[0] \
        + (-1*stokeslet_vec(rprime,zhat)-2*h*stokesletdipole_vec(rprime,zhat,zhat)+2*h**2*sourcedoublet_vec(rprime,zhat))*e[1]

    return vec

def approximate_blakelet(y,x,e):
    hvec=np.zeros_like(x)
    hvec[-1]=x[-1]
    h = x[-1]
    zhat=np.zeros_like(x)
    zhat[-1]=1
    r = y-x
    rprime = y-(x-2*hvec)
    ephi = np.zeros_like(x)
    ephi[0]=1

    yy= np.outer(y,y)
    yhyh = np.outer(y+2*hvec,y+2*hvec)
    xzzx = np.outer(ephi,zhat)-np.outer(zhat,ephi)

    vec = ((yy/np.dot(y,y)) - (1-2*np.dot(y,hvec)/np.dot(y,y))*yhyh/np.dot(y,y))@e


def theta_loop(t):
    return t%(2*np.pi)

def circ1(theta):
    return np.array([np.cos(theta),5.0+np.sin(theta)])

def circv(theta):
    return np.array([-np.sin(theta),np.cos(theta)])

def a1(t):
    return circ1(theta_loop(t-0.5*np.pi))

def av1(t):
    return circv(theta_loop(t-0.5*np.pi))

def G(r):     
    rsq = np.dot(r, r)
    if not(rsq==0):
        mat = np.eye(2)/np.sqrt(rsq) + np.multiply.outer(r,r)*rsq**(-3/2)
        return mat
    else:
        return np.eye(2)

# 
# def f1(t,x1,x2,x1dot,x2dot):
#     return np.dot(np.linalg.inv(np.eye(2)-G(x2-x1)),np.dot(np.linalg.inv(G(x2-x1)),x2dot)-np.dot(np.dot(G(x2-x1),G(x2-x1)),x1dot))

def update_vectorplot(n,Q,XX,YY,tt):
    XY = np.vstack([XX.ravel(),YY.ravel()]).T
    Flist = np.empty_like(XY)
    t=tt[n]

    for i in range(XY.shape[0]):
        Flist[i,:] = blakelet_vec(XY[i,:],a1(t),av1(t))
        Flist[i,:]=Flist[i,:]/np.linalg.norm(Flist[i,:])
    Q.set_UVC(Flist[:,0],Flist[:,1])
    return Q,


def get_F(t,z):
    F = (3/4)*a*blakelet_vec(z,a1(t),av1(t))
    return F

def plot_vfield():    
    xx = np.linspace(-100,100,20)
    yy = np.linspace(0,100,20)
    tt = np.linspace(0,2*np.pi,20)
    XX,YY = np.meshgrid(xx,yy) 
    XY = np.vstack([XX.ravel(),YY.ravel()]).T
    Flist = np.empty_like(XY)

    # active_1vals = circ1(theta_alternating(tt))
    # active_2vals = circ2(theta_alternating(tt-0.25))
    # active_v1vals = circv(theta_alternating(tt))
    # active_v2vals = circv(3*np.pi/4 - theta_alternating(tt-0.25))

    for i in range(XY.shape[0]):
        Flist[i,:] = blakelet_vec(XY[i,:],a1(tt[0]),av1(tt[0]))
        Flist[i,:]=Flist[i,:]/np.linalg.norm(Flist[i,:])
    fig, ax = pl.subplots(1,1)
    Q = ax.quiver(XY[:,0], XY[:,1], Flist[:,0], Flist[:,1], pivot='mid', color='k', units='inches')

    ax.set_xlim(-100, 100)
    ax.set_ylim(0, 100)
    anim=animation.FuncAnimation(fig,update_vectorplot,fargs=(Q,XX,YY,tt),interval=50,blit=False)
    fig.tight_layout()
    pl.show()

def plot_streamlines():
    xx = np.linspace(-20,20,50)
    yy = np.linspace(0.25,20,50)
    XX,YY = np.meshgrid(xx,yy) 
    XY = np.vstack([XX.ravel(),YY.ravel()]).T 
    t = np.linspace(0,2*np.pi,20)
    fig,ax=pl.subplots(1,1)
    u = np.empty_like(XX)
    v = np.empty_like(YY)
    for i in range(xx.size):
        for j in range(yy.size):
            z = [XX[j,i],YY[j,i]]
            s=si.solve_ivp(get_F,[0,2*np.pi],z,rtol=1e-10)
            u[j,i]=s.y[0,-1]-s.y[0,0]
            v[j,i]=s.y[1,-1]-s.y[1,0]
        print("Solving [{0}/{1}]".format(i+1,xx.size))

    r=(XX**2+(YY-5.0)**2)**0.5
    circle = r>=2.5
    u,v = np.where(circle,u,0), np.where(circle,v,0)
    stream = ax.streamplot(XX,YY,u,v,color= np.log10((u)**2+(v)**2)/2,density=2)
    ax.plot(a1(t)[0,:],a1(t)[1,:],'-r',linewidth=1.5,linestyle='dashed')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(stream.lines,cax=cax)
    
    ax.set_xlim(xx[0],xx[-1])
    ax.set_ylim(0,yy[-1])
    ax.set_aspect(1)


    pl.show()
    pl.close()

if __name__=="__main__":
    plot_streamlines()

    
        


