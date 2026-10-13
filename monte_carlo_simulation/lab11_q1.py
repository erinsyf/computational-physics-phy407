# Question 1
"""
Author: Erin Fung and Valerie MacIsaac

Explanatory Notes: This code explores example 10.3 of the textbook about Monte
Carlo simulations. It is an exercise in Monte Carlo simulations that is used
throughout the lab.
"""

# Import Libraries
import matplotlib.pyplot as plt
from random import random, randrange
from math import exp, pi
from numpy import ones, array, linspace
from scipy.optimize import curve_fit

# Parameters
def run(T, steps):
  " to test different T and step values "

  N = 1000 # number of particles

  # Create 2D array to store quantum numbers
  n = ones([N,3], int)

  # Main loop
  eplot = [] # Energies to plot
  E = 3*N*pi*pi/2

  for k in range(steps):
    # choose the particle and quantum number
    i = randrange(N) # random particle in N particles
    j = randrange(3) # random quantum number (nx, ny, nz) between 1 and 3
    if random()<0.5: # 50/50 chance of +1 or -1 change
      dn = 1
      dE = (2*n[i,j]+1)*pi*pi/2
    else:
      dn = -1
      dE = (-2*n[i,j]+1)*pi*pi/2

    # Decide whether to accept the move from equation 10.60
    if n[i,j]>1 or dn==1:
      if random()< exp(-dE/T):
        n[i,j] += dn
        E += dE

    eplot.append(E) # append the new energy

  # First plot the energy as a function of time (as in the textbook code)
  plt.figure(1)
  plt.plot(eplot)
  plt.ylabel('Energy')
  plt.xlabel('Step')
  plt.title(rf'Energy at Each Step for $k_bT = {T}$')
  plt.grid()
  plt.show()

  # This calculates the energy of each particle, neglecting constant factors
  energy_n = n[:, 0]**2 + n[:, 1]**2 + n[:, 2]**2
  # This calculates the frequency distribution and creates a plot
  plt.figure(2)
  plt.clf()
  hist_output = plt.hist(energy_n, 50) # frequency of each energy e_n in bin size of 50
  plt.xlabel(r'Energies $e_n = n_x^2 + n_y^2 + n_z^2$')
  plt.ylabel('Frequency of Energies')
  plt.title(rf'Histogram of Energies at Last Step for $k_bT = {T}$')
  plt.grid()
  plt.show()
  # This is the frequency distribution
  energy_frequency = hist_output[0]
  # This is what the x-axis of the plot should look like
  # if we plot the energy distribution as a function of n
  # the 2nd axis of hist_output contains the boundaries of the array. # Instead, we want their central value, below.
  energy_vals = 0.5*(hist_output[1][:-1] + hist_output[1][1:]) # bin centers
  n_vals = energy_vals**0.5 # n = sqrt(e_n)
  # Create the desired plot
  plt.figure(3)
  plt.clf()
  plt.bar(n_vals, energy_frequency, width=0.1)
  plt.xlabel(r'$n = \sqrt{n_x^2 + n_y^2 + n_z^2}$ (quantum number)')
  plt.ylabel(r'Frequency of Energies $e_n = n^2$')
  plt.title(rf'Energy Frequency vs Quantum number at Last Step for $k_bT = {T}$')
  plt.grid()
  plt.show()

  # PART ii.

  n_average_numerator = 0
  for i in range(len(energy_frequency)):
    n_average_numerator += energy_frequency[i] * n_vals[i]
  n_avg = n_average_numerator/sum(energy_frequency)

  return n_avg, E # return average n value and total energy E

# Test
kb_T_vals = [10, 40, 100, 400, 1200, 1600]

# test number of steps needed for each one
n_avg_0, E_0 = run(kb_T_vals[0], 100000)
n_avg_1, E_1 = run(kb_T_vals[1], 250000)
n_avg_2, E_2 = run(kb_T_vals[2], 250000)
n_avg_3, E_3 = run(kb_T_vals[3], 550000)
n_avg_4, E_4 = run(kb_T_vals[4], 3000000)
n_avg_5, E_5 = run(kb_T_vals[5], 3000000)

# Plot n(T) and E(T)
n_T = array([n_avg_0, n_avg_1, n_avg_2, n_avg_3, n_avg_4, n_avg_5])
E_T = array([E_0, E_1, E_2, E_3, E_4, E_5])

plt.figure(4)
plt.plot(kb_T_vals, n_T, 'o-')
plt.xlabel(r'$k_bT$')
plt.ylabel(r'$\overline{n}$')
plt.title('Average n(T)')
plt.grid()
plt.show()

plt.figure(5)
plt.plot(kb_T_vals, E_T, 'o-')
plt.xlabel(r'$k_bT$')
plt.ylabel(r'$E$')
plt.title('Total energy E (dimensionless)')
plt.grid()
plt.show()

# Calculate heat capacity deltaE/deltaT

def slope(x, a, b):
    return a*x+ b

params, *_ = curve_fit(slope, kb_T_vals, E_T, p0 = [1e6/600, 0.0])

print('deltaE/deltaT = ', params[0])

kbt = linspace(0, 1600)

#plot best fit against original

plt.figure(6)
plt.plot(kb_T_vals, E_T, 'o', label = 'Simpulated points ')
plt.plot(kbt, slope(kbt, *params), '-', label= 'Best Fit')
plt.xlabel(r'$k_bT$')
plt.ylabel(r'$E$')
plt.title('Total energy E (dimensionless)')
plt.legend()
plt.grid()
plt.show()

