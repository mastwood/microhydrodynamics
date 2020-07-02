#Solves the optimal control problem
"""
Created on 2020-06-25
@author: Michael Astwood
"""

from casadi import *

T=100 # time horizon
N=100 # control intervals
M=4  # rk4 steps per interval

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

G=(6*pi*(v1+dot(y-x1,v1)*(y-x1)/dot(y-x1,y-x1))/sqrt(dot(y-x1,y-x1)) \
    + 6*pi*(v2+dot(y-x2,v2)*(y-x2)/dot(y-x2,y-x2))/sqrt(dot(y-x2,y-x2)))/(8*np.pi)  #stokeslet

X=vertcat(theta1,theta2,y1,y2)  # configuration coordinates
ydot=G # equations of motion

Xdot=vertcat(u,ydot) # control system

# Fixed step Runge-Kutta 4 integrator
DT = T/N/M
f = Function('f', [X, u], [Xdot, L])
X0 = MX.sym('X0', 4) # arbitrary initial conditions
UU = MX.sym('U' , 2)
XX = X0 # initialize coordinates
Q = 0 # total cost
for j in range(M):
    print("Integrating... ",j)
    k1, k1_q = f(XX, UU)
    k2, k2_q = f(XX + DT/2 * k1, UU)
    k3, k3_q = f(XX + DT/2 * k2, UU)
    k4, k4_q = f(XX + DT * k3, UU)
    XX=XX+DT/6*(k1 +2*k2 +2*k3 +k4)
    Q = Q + DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q)
F = Function('F', [X0, UU], [XX, Q],['x0','p'],['xf','qf'])

# Start with an empty NLP
w=[]
w0 = []
lbw = []
ubw = []
J = 0
g=[]
lbg = []
ubg = []


# lift initial conditions
Xk = MX.sym('X0', 4)
w += [Xk]
lbw += [0, pi/2, 0, 50]
ubw += [0, pi/2, 0, 100]
w0  += [0, pi/2, 0, 75]

# generate multiple shooting constraints
for k in range(N):
    print("Generating problem... ",k)

    # control variables
    Uk = MX.sym('U_' + str(k),2)
    w += [Uk]
    lbw += [-100,-100]
    ubw += [ 100, 100]
    w0 += [0,0]

    # integrator
    Fk = F(x0=Xk, p=Uk)
    Xk_end = Fk['xf']
    J  = J+Fk['qf']

    # trajectory variables
    Xk = MX.sym('X_' + str(k+1), 4)
    w   += [Xk]
    lbw += [0,0,50,50]
    ubw += [2*np.pi,2*np.pi, 100, 100]
    w0  += [0,np.pi/2,100,100]

    # equality constraint
    g+=[Xk_end-Xk]
    lbg+=[0,0,0,0]
    ubg+=[0,0,0,0]

# boundary condition constraint
g+=[Xk[3]-90]
lbg+=[0]
ubg+=[0]
g+=[Xk[2]-90]
lbg+=[0]
ubg+=[0]

# Create an NLP solver
print("Setting up NLP solver...")
prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
solver = nlpsol('solver', 'ipopt', prob)

# Solve the NLP
print("Starting solver...")
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
w_opt = sol['x']

# Plot the solution
u_opt = w_opt
x_opt = [[0, pi/2, 5,0]]
for k in range(N):
    Fk = F(x0=x_opt[-1], p=u_opt[k])
    x_opt += [Fk['xf'].full()]
y1_opt = [r[2] for r in x_opt]
y2_opt = [r[3] for r in x_opt]
print(u_opt.shape)

tgrid = [T/N*k for k in range(N+1)]
import matplotlib.pyplot as plt
fig,(ax1,ax2)=plt.subplots(2)
ax1.plot(tgrid, y1_opt)
ax1.plot(tgrid, y2_opt)
ax1.set_xlabel('t')
ax1.grid()
ax1.legend(['y1','y2'])

ax2.step(tgrid, vertcat(DM.nan(1), u_opt[2::4]), '-.')
ax2.step(tgrid, vertcat(DM.nan(1), u_opt[3::4]), '-.')
ax2.set_xlabel('t')
ax2.legend(['u1','u2'])
ax2.grid()
plt.show()