#Solves the optimal control problem via indirect methods
"""
Created on 2020-06-26
@author: Michael Astwood
"""

from casadi import *
import numpy as NP
import matplotlib.pyplot as pl

# Define variables and common expressions

u1=MX.sym('u_1')
u2=MX.sym('u_2') 
u=vertcat(u1,u2) # control params = angular speed of active particle

theta1=MX.sym('theta_1')
theta2=MX.sym('theta_2')
theta=vertcat(theta1,theta2)

y1=MX.sym('y_1')
y2=MX.sym('y_2') 
y=vertcat(y1,y2) # passive params = position of passive particle

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

X=vertcat(theta1,theta2,y1,y2)  # configuration coordinates
Xdot=vertcat(u,ydot) # control system

lam=MX.sym('lambda',4) # lagrange multipliers
H=dot(lam,Xdot)+L # Hamiltonian

ldot=-gradient(H,X)
print("Hamiltonian: ", H)


