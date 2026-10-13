# Question 1: Solution to Newman 8.8, Space garbage.

"""
Author: Erin Fung and Valerie MacIsaac

Explanatory Note: This code implements an adaptive step size fourth-order
Runge-Kutta method to solve the problem presented in Newman Exercise 8.8.
"""

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import time # for timing the algorithms

# Define Plot Style
plt.style.use('default')
plt.rcParams['figure.dpi'] = 300 # set higher resolution figures


# Defining function that returns right hand side of equation
def rhs(r):
    """ The right-hand-side of the equations
    INPUT:
    r = [x, vx, y, vy] are floats (not arrays)
    note: no explicit dependence on time
    OUTPUT:
    1x2 numpy array, rhs[0] is for x, rhs[1] is for vx, etc"""
    M = 10.
    L = 2.

    x = r[0]
    vx = r[1]
    y = r[2]
    vy = r[3]

    r2 = x**2 + y**2
    Fx, Fy = - M * np.array([x, y], float) / (r2 * np.sqrt(r2 + .25*L**2))
    return np.array([vx, Fx, vy, Fy], float)

# Apply 4th Runge-Kutta Method
# Initialising step variable
a = 0.0
b = 10.0
N = 10000 # to be comparable with the adaptive method
h_norm = (b-a)/N
tpoints = np.arange(a, b, h_norm)
# Storing values
xpoints = []
ypoints = []

# Below: ordering is x, dx/dt, y, dy/dt initial conditions
r = np.array([1., 0., 0., 1.], float)

start_norm = time.perf_counter()
for t in tpoints:
    xpoints.append(r[0])
    ypoints.append(r[2])
    k1 = h_norm*rhs(r)  # all the k's are vectors
    k2 = h_norm*rhs(r + 0.5*k1)  # note: no explicit dependence on time of the RHSs
    k3 = h_norm*rhs(r + 0.5*k2)
    k4 = h_norm*rhs(r + k3)
    r += (k1 + 2*k2 + 2*k3 + k4)/6
end_norm = time.perf_counter()

time_norm = end_norm - start_norm

# Adaptive Runge-Kutta
# We compare one step of size H=2h vs two steps of size h.
t = 0.0
delta = 1e-6     # target error per unit time
h = 0.01         # initial substep
r_adapt = np.array([1., 0., 0., 1.], float)

# Initialise lists with initial conditions to store adapted data points
xad = [r_adapt[0]]
yad = [r_adapt[2]]
t_adapt = [0] # store time steps

start_adapt = time.perf_counter()
while t < b:
    # adjust h to not overshoot final time with the composite 2h step
    H = 2.0*h
    if t + H > b:
        H = b - t
        h = 0.5*H

    # two half-steps of size h
    k1 = h * rhs(r_adapt)
    k2 = h * rhs(r_adapt + 0.5*k1)
    k3 = h * rhs(r_adapt + 0.5*k2)
    k4 = h * rhs(r_adapt + k3)
    r1_half = r_adapt + (k1 + 2*k2 + 2*k3 + k4)/6.0

    k1 = h * rhs(r1_half)
    k2 = h * rhs(r1_half + 0.5*k1)
    k3 = h * rhs(r1_half + 0.5*k2)
    k4 = h * rhs(r1_half + k3)
    r1 = r1_half + (k1 + 2*k2 + 2*k3 + k4)/6.0

    # one full step of size H = 2h
    k1 = H * rhs(r_adapt)
    k2 = H * rhs(r_adapt + 0.5*k1)
    k3 = H * rhs(r_adapt + 0.5*k2)
    k4 = H * rhs(r_adapt + k3)
    r2 = r_adapt + (k1 + 2*k2 + 2*k3 + k4)/6.0

    # Calculate the error epsilons
    eps_x = (r1[0] - r2[0]) / 30.0
    eps_y = (r1[2] - r2[2]) / 30.0
    eps = np.sqrt(eps_x**2 + eps_y**2)

    # Calculate rho
    rho = (H * delta) / eps

    if rho >= 1.0: # timestep too small, but still accurate
        # advance to the more accurate solution r1, append solution to xad, yad
        # and increase step
        r_adapt = r1
        t += H
        xad.append(r_adapt[0])
        yad.append(r_adapt[2])
        t_adapt.append(t)
        # try enlarging step modestly
        h = h * rho**0.25
    else:
        # reject; shrink step and retry without advancing
        h = h * rho**0.25
end_adapt = time.perf_counter()

time_adapt = end_adapt - start_adapt

# Plot the comparison
plt.figure()
plt.plot(xpoints, ypoints, 'o', ms = .5, label = 'Non-adaptive')
plt.plot(xad, yad, 'o', ms = 1.5, label = 'Adaptive')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title('Trajectory of a ball bearing around a space rod')
plt.grid()
plt.tight_layout()
plt.show()

# Print the Time Difference
print(f'Timing for non-adaptive RK4 = {time_norm}')
print(f'Timing for adaptive RK4 = {time_adapt}')

# Plot the Time-Steps for the adaptive method
dtpoints = np.array(t_adapt[1:]) - np.array(t_adapt[:-1])
plt.plot(t_adapt[:-1], dtpoints, '-', label = 'Adaptive Timesteps')
plt.legend()
plt.xlabel('Time t (s)')
plt.ylabel('Adaptive timesteps $h$ (s)')
plt.grid()
plt.title('Size of Time step as Function of Time')
