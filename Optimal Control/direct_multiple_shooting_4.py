#Solves the optimal control problem
"""
Created on 2020-06-25
@author: Michael Astwood
"""

from casadi import *

T=MX.sym('T') # time horizon
N=100 # control intervals
M=8  # rk4 steps per interval

# Define variables and common expressions

y=MX.sym('y',2,1)
x1=MX.sym('x1',2,1)
x2=MX.sym('x2',2,1)
v1=MX.sym('v1',2,1)
v2=MX.sym('v2',2,1)
u=vertcat(v1,v2)

L= dot(u,u) # cost function (proportional to energy expended)

#G=(3/4)*((v1+dot(y-x1,v1)*(y-x1)/dot(y-x1,y-x1))/sqrt(dot(y-x1,y-x1)) \
#    + (v2+dot(y-x2,v2)*(y-x2)/dot(y-x2,y-x2))/sqrt(dot(y-x2,y-x2)))  #stokeslet
eps=1
G=(3/4)*((v1+2*eps**2+dot(y-x1,v1)*(y-x1)/(dot(y-x1,y-x1)+eps**2))/sqrt(dot(y-x1,y-x1)+eps**2) \
   + (v2+2*eps**2+dot(y-x2,v2)*(y-x2)/(dot(y-x2,y-x2)+eps**2))/sqrt(dot(y-x2,y-x2)+eps**2))  #stokeslet
X=vertcat(x1,x2,y)  # configuration coordinates
ydot=G # equations of motion

Xdot=vertcat(v1,v2,ydot) # control system

# Fixed step Runge-Kutta 4 integrator
DT = T/N/M
f = Function('f', [X, u], [Xdot, L])
X0 = MX.sym('X0', 6) # arbitrary initial conditions
UU = MX.sym('U' , 4)
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

w+=[T]
lbw+=[1]
ubw+=[1]
w0+=[1]

# lift initial conditions
Xk = MX.sym('X0', 6)
w += [Xk]
lbw += [-100,-100,-100,-100, -10, -10]
ubw += [100,100,100,100, 10, 10]
w0  += [0, 1,1,0, 3, 3]
# control variables
Uk = MX.sym('U0',4)
w += [Uk]
lbw += [-100,-100,-100,-100]
ubw += [ 100, 100,100,100]
w0  += [0,0,0,0]
# generate multiple shooting constraints
for k in range(N):
    print("Generating problem... ",k)
    # integrator
    Fk = F(x0=Xk, p=Uk)
    Xk_end = Fk['xf']
    J  = J+Fk['qf']

    # equality constraint
    g+=[Xk_end-Xk]
    lbg+=[0,0,0,0,0,0]
    ubg+=[0,0,0,0,0,0]

    # trajectory variables
    Xk  = MX.sym('X_' + str(k+1),6)
    w   += [Xk]
    lbw += [-100,-100,-100,-100,-10,-10]
    ubw += [100,100,100,100,10, 10]
    w0  += [0,1,1,0,3,3]
    # control variables
    Uk = MX.sym('U_' + str(k),4)
    w += [Uk]
    lbw += [-100,-100,-100,-100]
    ubw += [ 100, 100,100,100]
    w0  += [0,0,0,0]

    g+=[Xk[0]**2+Xk[1]**2-1]
    lbg+=[0]
    ubg+=[0]
    g+=[Xk[2]**2+Xk[3]**2-1]
    lbg+=[0]
    ubg+=[0]
    g+=[Xk[0]*Uk[0]+Xk[1]*Uk[1]]
    lbg+=[0]
    ubg+=[0]
    g+=[Xk[2]*Uk[2]+Xk[3]*Uk[3]]
    lbg+=[0]
    ubg+=[0]



# boundary condition constraint
g+=[Xk[4]-4]
lbg+=[0]
ubg+=[0]
g+=[Xk[5]-4]
lbg+=[0]
ubg+=[0]

# Create an NLP solver
print("Setting up NLP solver...")
prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
solver = nlpsol('solver', 'ipopt', prob)

# Solve the NLP
print("Starting solver...")
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
u_opt = sol['x'][1::]

# # Plot the solution
# x_opt = [[0,1,1,0,3,3]]
# for k in range(N):
#     Fk = F(x0=x_opt[-1], p=u_opt[k*10+6:k*10+10])
#     x_opt += [Fk['xf'].full()]
y1_opt = u_opt[4::10] #[r[4] for r in x_opt]
y2_opt = u_opt[5::10] #[r[5] for r in x_opt]

tgrid = [sol['x'][0]/N*k for k in range(N+1)]
import matplotlib.pyplot as plt
fig,(ax1,ax2,ax3)=plt.subplots(3)
ax1.plot(tgrid, y1_opt)
ax1.plot(tgrid, y2_opt)
ax1.set_xlabel('t')
ax1.grid()
ax1.legend(['y1','y2'])

ax2.plot(tgrid, u_opt[6::10])
ax2.plot(tgrid, u_opt[7::10])
ax2.plot(tgrid, u_opt[8::10])
ax2.plot(tgrid, u_opt[9::10])

ax2.set_xlabel('t')
ax2.legend(['u1','u2','u3','u4'])

ax3.plot(u_opt[0::10], u_opt[1::10])
ax3.plot(u_opt[2::10], u_opt[3::10])
ax3.legend(['x1','x2'])

ax2.grid()
plt.show()