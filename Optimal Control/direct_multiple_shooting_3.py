"""
Created on 2020-06-26
@author: Michael Astwood
"""

from casadi import *

opti = Opti()

T=10
N=20 # control intervals
M=8  # rk4 steps per interval

X = opti.variable(4,N+1)

y1 = X[0,:]
y2 = X[1,:]
y=vertcat(y1,y2)

theta1=X[2,:]
theta2=X[3,:]
theta=vertcat(theta1,theta2)

U=opti.variable(2,N)
u1=U[0,:]
u2=U[1,:]
u=vertcat(u1,u2)

x11=cos(theta1)
x12=sin(theta1)
x21=cos(theta2)
x22=sin(theta2) 
x1=vertcat(x11,x12)
x2=vertcat(x21,x22) # explicit position of active particles

v11=-u1*sin(theta1)
v12=u1*cos(theta1)
v21=-u2*sin(theta2)
v22=u2*cos(theta2) 
v1=vertcat(v11,v12)
v2=vertcat(v21,v22) # explicit velocity of active particles

L= dot(v1,v1)+dot(v2,v2) # cost function (proportional to energy expended)

ydot=6*pi*(v1+dot(y-x1,v1)*(y-x1)/dot(y-x1,y-x1))/sqrt(dot(y-x1,y-x1)) \
    + 6*pi*(v2+dot(y-x2,v2)*(y-x2)/dot(y-x2,y-x2))/sqrt(dot(y-x2,y-x2))  #stokeslet
Xdot=vertcat(ydot,u)

opti.minimize(L)
f = Function('f', [X, u], [Xdot, L])

dt=T/N 
for k in range(N):
    print("Integrating... ",j)
    k1 = f(X[:,k], U[:,k])
    k2 = f(X[:,k] + dt/2 * k1, U[:,k])
    k3 = f(X[:,k] + dt/2 * k2, U[:,k])
    k4 = f(X[:,k] + dt * k3, U[:,k])
    X_next=X[:,k] + dt/6*(k1 +2*k2 +2*k3 +k4)
    opti.subject_to(X[:,k+1]==X_next)

opti.subject_to(X[:,-1]==[0,50])

opti.set_initial(y,[0,5])
opti.set_initial(theta,[0,pi/2])
opti.subject_to(fabs(theta1-theta2)>=0.001)

opti.solver('ipopt')
sol=opti.solve()
print(sol.value(y))
