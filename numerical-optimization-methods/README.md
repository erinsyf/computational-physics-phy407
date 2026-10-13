# Numerical Optimisation Methods

Python implementations of numerical optimisation and adaptive integration methods developed during PHY407 (Computational Physics) at the University of Toronto.

## Projects

### 1. Simulated Annealing — Lab 11, Question 3

**File:** `lab11_q3.py`

Explores simulated annealing applied to two optimisation problems:

- **Travelling Salesman Problem:** Finding shorter routes between randomly positioned cities using probabilistic acceptance and exponential cooling.
- **Function Minimisation:** Finding minima of two-dimensional functions using simulated annealing and Gaussian random sampling.

The simulations investigate the effects of cooling rates, random seeds, and optimisation constraints.

### 2. Adaptive Runge–Kutta — Lab 07, Question 1

**File:** `lab07_q1.py`

Implements an adaptive fourth-order Runge–Kutta method to simulate the motion of a particle orbiting a massive rod.

The simulation explores:

- Fixed versus adaptive time-stepping
- Numerical error estimation and step-size adjustment
- Computational accuracy and execution time
- Relationship between orbital dynamics and the required integration step size

## Requirements

- Python 3
- NumPy
- Matplotlib

## Acknowledgements

These projects were developed during PHY407: Computational Physics at the University of Toronto (2025), taught by Professor Nicolas Grisouard.

The problem formulations and numerical methods are based on PHY407 course materials and Mark Newman's *Computational Physics*, including Exercises 8.8 and 10.10 and the textbook's simulated annealing examples.

The Python implementations were written by Erin Fung for coursework completed collaboratively with Valerie MacIsaac. The projects preserve the original undergraduate coursework implementations.
