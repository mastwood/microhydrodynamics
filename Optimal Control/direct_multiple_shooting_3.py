"""
Created on 2020-06-26
@author: Michael Astwood
"""

from casadi import *

opti = Opti()

T=10
N=20 # control intervals
M=8  # rk4 steps per interval

U=opti.variable(4,N)
X=opti.variable(6,N)
x1=X[[0,1],:]
x2=X[[2,3],:]
y =X[[4,5],:]
v1=U[[0,1],:]
v2=U[[2,3],:]

L= (3/4)*dot(U,U) # cost function (proportional to energy expended)

ydot=(3/4)*(v1+dot(y-x1,v1)*(y-x1)/dot(y-x1,y-x1))/sqrt(dot(y-x1,y-x1)) \
    + (3/4)*(v2+dot(y-x2,v2)*(y-x2)/dot(y-x2,y-x2))/sqrt(dot(y-x2,y-x2))  #stokeslet
Xdot=vertcat(v1,v2,ydot)

f = Function('f', [X, U], [Xdot,L]) # differential equation RHS plus cost

dt=T/N
Q=0 
for k in range(N):
    print("Integrating... ",k)
    k1,k1_q = f(X, U)
    k2,k2_q = f(X + dt/2 * k1, U)
    k3,k3_q = f(X + dt/2 * k2, U)
    k4,k4_q = f(X + dt * k3, U)
    Q = Q + dt/6*(k1_q + 2*k2_q + 2*k3_q + k4_q)
    X_next=X[:,k] + dt/6*(k1[:,k] +2*k2[:,k] +2*k3[:,k] +k4[:,k])
    if not(k==N-1):
        opti.subject_to(X[:,k+1]==X_next) #node conditions
opti.minimize(Q) # total cost
opti.subject_to(X[[4,5],-1]==[4,4]) #final conditions
opti.subject_to(y[:,0]==[3,3]) #initial conditions
opti.subject_to(x1[:,0]==[0,1]) #initial conditions
opti.subject_to(x2[:,0]==[1,0]) #initial conditions

opti.solver('ipopt')
sol=opti.solve()
print(sol.value(y))
