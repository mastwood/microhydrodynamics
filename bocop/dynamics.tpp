// Function for the dynamics of the problem
// dy/dt = dynamics(y,u,z,p)

// The following are the input and output available variables 
// for the dynamics of your optimal control problem.

// Input :
// time : current time (t)
// normalized_time: t renormalized in [0,1]
// initial_time : time value on the first discretization point
// final_time : time value on the last discretization point
// dim_* is the dimension of next vector in the declaration
// state : vector of state variables
// control : vector of control variables
// algebraicvars : vector of algebraic variables
// optimvars : vector of optimization parameters
// constants : vector of constants

// Output :
// state_dynamics : vector giving the expression of the dynamic of each state variable.

// The functions of your problem have to be written in C++ code
// Remember that the vectors numbering in C++ starts from 0
// (ex: the first component of the vector state is state[0])

// Tdouble variables correspond to values that can change during optimization:
// states, controls, algebraic variables and optimization parameters.
// Values that remain constant during optimization use standard types (double, int, ...).

// def regularized_stokeslet(y1,y2,x11,x12,x21,x22,v11,v12,v21,v22):
//     e=0.1
//     r11=y1-x11
//     r12=y2-x12
//     r21=y1-x21
//     r22=y2-x22

//     r1v1=r11*v11+r12*v12
//     r2v2=r21*v21+r22*v22

//     r1r1=r11**2+r12**2
//     r2r2=r21**2+r22**2

//     rinvsq1=1/(r1r1+e**2)
//     rinvsq2=1/(r2r2+e**2)

//     ydot1=(rinvsq1**(3/2))*(r1r1*v11+2*(e**2)*v11+r1v1*r11) \
//         + (rinvsq2**(3/2))*(r2r2*v21+2*(e**2)*v21+r2v2*r21)
//     ydot2=(rinvsq1**(3/2))*(r1r1*v12+2*(e**2)*v12+r1v1*r12) \
//         + (rinvsq2**(3/2))*(r2r2*v22+2*(e**2)*v22+r2v2*r22)
//     return 0.1*(3/4)*np.array([ydot1,ydot2])
#include "header_dynamics"
{
	// HERE : description of the function for the dynamics
	// Please give a function or a value for the dynamics of each state variable
	r=constants[0];
	double e=0.1;
	y_1=state[0];
	y_2=state[1];
	x_1=state[2];
	x_2=state[3];
	xdot_1=control[0];
	xdot_2=control[1];
	y_dot1=algebraic[0];
	y_dot2=algebraic[1];
	r_1=y_1-x_1;
	r_2=y_2-x_2;
	rr=(r_1*r_1+r_2*r_2);
	rinvsq=pow(rr,-1);
	rv=r_1*xdot_1+r_2*xdot_2;
	
	state_dynamics[0] = pow(rinvsq,3/2)*(rr*xdot_1+2*e**2*xdot_1+rv*r_1)
	state_dynamics[1] = pow(rinvsq,3/2)*(rr*xdot_2+2*e**2*xdot_2+rv*r_2)
	state_dynamics[2] = xdot_1;
	state_dynamics[3] = xdot_2;
	state_dynamics[4] = xdot_1*xdot_1+xdot_2*xdot_2;
}


