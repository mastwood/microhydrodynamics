import numpy as np
import scipy.integrate as si
import matplotlib.pyplot as pl 
from matplotlib import animation 
#from numba import jit, njit, vectorize

a=0.1

#@njit
def stokeslet_vec(x, e):
    # Spagnolie & Lauga 2012 notation
    rinvsq = 1.0/np.dot(x, x)
    rinv = 1.0/np.linalg.norm(x)
    
    vec = rinv*(e + np.dot(x,e)*x*rinvsq)
    
    return vec

#@njit
def stokesletdipole_vec(x, d, e):
    # Spagnolie & Lauga 2012 notation
    rinvsq = 1.0/np.dot(x, x)
    rinv = 1.0/np.linalg.norm(x)
    
    vec = rinvsq*(rinv*(np.dot(d,x)*e - np.dot(e,x)*d - np.dot(d,e)*x) +
                  3.0*np.dot(e,x)*np.dot(d,x)*x*rinv*rinvsq)
    
    return vec

#@njit
def sourcedoublet_vec(x, e):
    # Spagnolie & Lauga 2012 notation
    rinvsq = 1.0/np.dot(x, x)
    rinv = 1.0/np.linalg.norm(x)

    vec = rinv*rinvsq*(-e + 3.0*np.dot(x,e)*x*rinvsq)
    
    return vec

#@njit
def blakelet_vec(x,y,e,h):
    # Spagnolie & Lauga 2012 notation
    # No slip plane boundary at z=0. h is z co-ordinate of stokeslet
    # x = position of active particle
    # y = position of passive particle
    # e = direction of motion of active particle
    hvec=np.zeros_like(x)
    hvec[-1]=x[-1]
    zhat=np.zeros_like(x)
    zhat[-1]=1
    r = y-x
    rprime = y-(x-2*hvec)
    ephi = np.zeros_like(x)
    ephi[0]=1

    vec = (stokeslet_vec(r,ephi)-stokeslet_vec(rprime,ephi)+2*h*stokesletdipole_vec(rprime,ephi,zhat)-2*h**2*sourcedoublet_vec(rprime,ephi))*e[0] \
        + (stokeslet_vec(r,ephi)-stokeslet_vec(rprime,zhat)-2*h*stokesletdipole_vec(rprime,zhat,zhat)+2*h**2*sourcedoublet_vec(rprime,zhat))*e[1]

    return vec

#@njit
def theta_alternating(t):
    if t%1<=0.25:
        return 2*np.pi*(t%1)
    elif 0.25<t%1 and t%1<=0.5:
        return np.pi/4
    elif 0.5<t%1 and t%1<=0.75:
        return np.pi/4 - 2*np.pi*(t%1-0.5)
    elif 0.75<t%1 and t%1<=1.0:
        return 0

#@njit
def circ1(theta):
    return np.array([-2.0 + np.cos(theta),np.sin(theta)])

#@njit
def circ2(theta):
    return np.array([2.0 + np.cos(theta),np.sin(theta)])

#@njit
def circv(theta):
    return np.array([-np.sin(theta),np.cos(theta)])

#@njit
def a1(t):
    return circ1(theta_alternating(t))

#@njit
def a2(t):
    return circ2(theta_alternating(t-0.25))

#@njit
def av1(t):
    return circv(theta_alternating(t))

#@njit
def av2(t):
    return circv(3*np.pi/4 - theta_alternating(t-0.25))

#@njit
def G(r):     
    rsq = np.dot(r, r)
    if not(rsq==0):
        mat = np.eye(2)/np.sqrt(rsq) + np.multiply.outer(r,r)*rsq**(-3/2)
        return mat
    else:
        return np.eye(2)

#@njit
def f1(t,x1,x2,x1dot,x2dot):
    return np.dot(np.linalg.inv(np.eye(2)-G(x2-x1)),np.dot(np.linalg.inv(G(x2-x1)),x2dot)-np.dot(np.dot(G(x2-x1),G(x2-x1)),x1dot))

def update_vectorplot(n,Q,XX,YY,aa1,aa2,aav1,aav2):
    XY = np.vstack([XX.ravel(),YY.ravel()]).T
    Flist = np.empty_like(XY)
    tt = np.linspace(0,1,20)
    t=tt[n]

    for i in range(XY.shape[0]):
        Flist[i,:] = np.dot(G(XY[i]-aa1[:,n]), f1(t,aa1[:,n],aa2[:,n],aav1[:,n],aav2[:,n]))\
            +np.dot(G(XY[i]-aa2[:,n]), f1(t,aa1[:,n],aa2[:,n],aav2[:,n],aav1[:,n]))
        Flist[i,:]=Flist[i,:]/np.linalg.norm(Flist[i,:])
    Q.set_UVC(Flist[:,0],Flist[:,1])
    return Q,

#@njit
def get_F(t,z):
    F = np.dot(G(z-a1(t)), f1(t,a1(t),a2(t),av1(t),av2(t)))\
            +np.dot(G(z-a2(t)), f1(t,a1(t),a2(t),av2(t),av1(t)))
    return F

def plot_vfield():    
    xx = np.linspace(-100,100,20)
    yy = np.linspace(-100,100,20)
    tt = np.linspace(0,1,20)
    XX,YY = np.meshgrid(xx,yy) 
    XY = np.vstack([XX.ravel(),YY.ravel()]).T
    Flist = np.empty_like(XY)

    active_1vals = circ1(theta_alternating(tt))
    active_2vals = circ2(theta_alternating(tt-0.25))
    active_v1vals = circv(theta_alternating(tt))
    active_v2vals = circv(3*np.pi/4 - theta_alternating(tt-0.25))

    for i in range(XY.shape[0]):
        Flist[i,:] = np.dot(G(XY[i,:]-active_1vals[:,0]), f1(0,active_1vals[:,0],active_2vals[:,0],active_v1vals[:,0],active_v2vals[:,0]))\
            +np.dot(G(XY[i,:]-active_2vals[:,0]), f1(0,active_1vals[:,0],active_2vals[:,0],active_v2vals[:,0],active_v1vals[:,0]))
        Flist[i,:]=Flist[i,:]/np.linalg.norm(Flist[i,:])
    fig, ax = pl.subplots(1,1)
    Q = ax.quiver(XY[:,0], XY[:,1], Flist[:,0], Flist[:,1], pivot='mid', color='r', units='inches')

    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    anim=animation.FuncAnimation(fig,update_vectorplot,fargs=(Q,XX,YY,active_1vals,active_2vals,active_v1vals,active_v2vals),interval=50,blit=False)
    fig.tight_layout()
    pl.show()

if __name__=="__main__":
    xx = np.linspace(-100,100,8)
    yy = np.linspace(-100,100,8)
    XX,YY = np.meshgrid(xx,yy) 
    XY = np.vstack([XX.ravel(),YY.ravel()]).T
    #solns=np.empty((XY.shape[0]))

    fig,ax=pl.subplots(1,1)
    disp=np.empty_like(XY)
    for i in range(XY.shape[0]):
        s=si.solve_ivp(get_F,[0,1],XY[i,:],rtol=1e-10)
        disp[i,:]=s.y[:,-1]-s.y[:,0]
        print("Solving [{0}/{1}]".format(i+1,XY.shape[0]))
    ax.quiver(XY[:,0],XY[:,1],disp[:,0],disp[:,1])

    pl.show()


    
        


