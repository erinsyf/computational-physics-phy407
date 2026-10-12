# Quantum Dynamics Simulations

Python implementations of numerical methods for solving the time-independent and time-dependent Schrödinger equations, developed during PHY407 (Computational Physics) at the University of Toronto.

## Projects

### 1. Quantum Energy Eigenstates

**File:** `stationary_states/lab07_q2.py`

Numerical solutions of the time-independent Schrödinger equation for harmonic and anharmonic oscillators.

**Methods:**
- Fourth-order Runge–Kutta (RK4)
- Shooting and secant methods for energy eigenvalues
- Simpson's rule for wavefunction normalisation
- Comparison with analytical harmonic-oscillator solutions

### 2. Quantum Wavefunction Time Evolution

**File:** `time_evolution/lab09_q2.py`

Numerical simulation of wavefunction evolution under different potentials using the time-dependent Schrödinger equation.

**Methods:**
- Finite-difference Hamiltonian
- Crank–Nicolson time integration
- Infinite square well, harmonic oscillator, and double-well potentials
- Wavefunction normalisation, energy and position diagnostics
- Wavefunction visualisation and animation

## Requirements to run code

- Python 3
- NumPy
- SciPy
- Matplotlib

## Acknowledgements

These projects were developed as part of PHY407: Computational Physics at the University of Toronto (2025), taught by Professor Nicolas Grisouard.

The numerical methods and problem formulations are based on course materials and exercises from Mark Newman's *Computational Physics*, particularly Example 8.9, Exercise 8.14, and Exercise 9.8.

The Python implementations were written by Erin Fung for coursework completed collaboratively with Valerie MacIsaac.

The projects are shared for educational and portfolio purposes and do not claim originality of the underlying numerical methods.
