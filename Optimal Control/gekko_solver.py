from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as pl

g=GEKKO(remote=False)

nt=101
g.time=np.linspace(0,10,nt)

u1=g.MV(value=0)
u2=g.MV(value=0)

y1=g.CV(value=5)
y2=g.CV(value=0)

y1f=g.FV()
y2f=g.FV()

g.Equation(y1f==50)
g.Equation(y2f==-50)
g.Connection(y1f,y1,pos2='end')
g.Connection(y2f,y2,pos2='end')

theta1=g.Var(value=0)
theta2=g.Var(value=0)

g.Equation(u1==theta1.dt())
g.Equation(u2==theta2.dt())

#ydot=6*pi*(v1+dot(y-x1,v1)*(y-x1)/dot(y-x1,y-x1))/sqrt(dot(y-x1,y-x1)) \
#    + 6*pi*(v2+dot(y-x2,v2)*(y-x2)/dot(y-x2,y-x2))/sqrt(dot(y-x2,y-x2)) 

def stokeslet(y1,y2,theta1,theta2,u1,u2):
    k=6*np.pi 
    v11=u1*g.sin(theta1)
    v12=-u1*g.cos(theta1)
    v21=u2*g.sin(theta2)
    v22=-u1*g.cos(theta2)
    x11=g.cos(theta1)
    x12=g.sin(theta1)
    x21=g.cos(theta2)
    x22=g.sin(theta2)

    d1=(y1-x11)*v11+(y2-x12)*v12
    d2=(y1-x11)**2+(y2-x12)**2
    d3=g.sqrt(d2)

    d4=(y1-x21)*v21+(y2-x12)*v22
    d5=(y1-x21)**2+(y2-x22)**2
    d6=g.sqrt(d5)

    ydot1=(k*(v11+d1*(y1-x11))/d2)/d3+(k*(v21+d4*(y1-x21))/d5)/d6
    ydot2=(k*(v12+d1*(y2-x12))/d2)/d3+(k*(v22+d4*(y2-x22))/d5)/d6
    return [ydot1,ydot2]

g.Equation(y1.dt()==stokeslet(y1,y2,theta1,theta2,u1,u2)[0])
g.Equation(y2.dt()==stokeslet(y1,y2,theta1,theta2,u1,u2)[1])

J=g.Var(value=0)
Jf=g.FV()

g.Equation(J.dt()==u1**2+u2**2)

g.Connection(Jf,J,pos2='end')
g.Obj(Jf)

g.options.IMODE = 6  # optimal control
g.options.NODES = 3  # collocation nodes
g.options.SOLVER = 3 # solver (IPOPT)

g.solve(disp=False,debug=0) # Solve