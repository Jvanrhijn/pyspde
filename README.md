# pyspde

`pyspde` is a very simple library for solving stochastic partial differential equations
in Python. The basic functionality is largely based on [`xSPDE`](https://github.com/peterddrummond/xspde_matlab).
The functionality of `pyspde` is very limited in comparison. 

## Features

* Solving stochastic equations of the form `da/dt = d^2a/dx^2 + A(a) + B(a)w(t)`, 
   where `a : R -> R`, `A` is an arbitrary function of `a` and its first spatial derivative,
   and `B` is an arbitrary function of `a`. The noise function `w(t)` is delta-correlated,
   and can be user-provided (see `src/noises.py` for an example implementation).
* Solving stochastic ordinary differential equations via the same interface.
* Basic error analysis and extrapolation of results. The methods used are the same as in
  `xSPDE`.