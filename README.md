Research in microhydrodynamics and control theory.

# Dynamics Solvers
- field_lines.py computes the field lines for various stokes singularities
- old_stokes.py is an implementation of the stokeslet 
- quaternions.py is a file written by my advisor which implements a number of quaternion functions
- stokes_ode.py implements a number of functions which make computing stokeslet solutions quicker
- StokesSingularities.py implements basic stokes singularity Greens' functions

# Optimal Control
- direct_multiple_shooting.py implements the multiple direct shooting algorithm using CasADi
- direct_multiple_shooting_dae.py is an alternative implementation of direct multiple shooting
- gekko_solver.py and gekko_solver_2.py implement direct orthogonal collocation using GEKKO
- indirect_single_shooting.py is an attempt at the Pontryagin Minumum Principle as applied to this control problem

# Symbolics
- mostly mathematica files, containing solvers and plotters for the perturbation problem and for the full problem
