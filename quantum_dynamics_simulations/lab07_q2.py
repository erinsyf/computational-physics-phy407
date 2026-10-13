# Question 2: Quantum Oscillators

"""
Author: Erin Fung and Valerie MacIsaac

Explanatory Notes: This exercise involves numerically solving the
time-independent Schrödinger equation using the fourth-order Runge-Kutta method.
The solution focuses on calculating and plotting the wavefunctions for the first
three discrete energy states in both a harmonic and an anharmonic potential well.
"""

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
plt.rcParams['figure.dpi'] = 300 # set higher resolution figures


# Defining constants used throughout problem
hbar = const.hbar # Reduced Planck's constant
e = const.e #elementary charge
m = const.m_e # electron mass
a = 1e-11 # meters
V_0 = 50 * e # in Joules
N = 1000

# Question 2a): Modifying Example 8.9 for quadratic potential

# Defining potential function

def V_quad(x):
  """
  Returns the value of the quadratic potential in a harmonic oscillator,
  at position x.
  Input:
  x = Position in space of electron (float)
  Output:
  V = Value of the quadratic potential (float)
  """
  return V_0 * x**2/a**2

def f(r, x, E, V):
  """
  Calculates the derivative of variable r = [psi, phi], where
  psi(x) is the solution to the Schrodinger's equation for some potential V.
  Input:
  r = [psi, phi] list of variable values
  x = position (float) where the wavefunction is evaluated
  E = Energy in J (float)
  V = potential function
  Output:
  f = [fpsi, fphi] list of derivative values
  """
  psi = r[0]
  phi = r[1]
  fpsi = phi
  fphi = (2*m/hbar**2)*(V(x) - E)*psi
  return np.array([fpsi, fphi], float)

# Calculate the wavefunction for some energy E
def solve(E, V, x0, x1):
  """
  Calculate the solution psi with energy E using the 4th order Runge-Kutta method
  with dr/dt = f(r, x, E, V) for some potential V(x).
  The solution is found between x0 and x1.
  Keep track of each r value at each x value iterated over.
  Input:
  E = Energy in J (float)
  V = potential function
  x-range = [x0, x1] (float)
  Output:
  psi(x) = Solution to the Schrodinger's equation in the range [x0, x1]
  """
  psi = 0.0
  phi = 1.0 # define initial dpsi/dx = phi as 1.0, as this doesn't matter
  h = (x1 - x0)/N
  r = np.array([psi, phi], float)
  psi_x = []
  # Runge - kutta to find r = (psi, phi) for x in
  for x in np.arange(x0, x1, h):
    psi_x.append(r[0])
    k1 = h*f(r, x, E, V)
    k2 = h*f(r + 0.5*k1, x+0.5*h, E, V)
    k3 = h*f(r + 0.5*k2, x+0.5*h, E, V)
    k4 = h*f(r + k3, x+h, E, V)
    r += (k1 + 2*k2 + 2*k3 + k4)/6

  return psi_x

# Find energy using secant method

def secant(e1, e2, target, V, x0, x1):
    """
    Find energy such that psi(xR; E) ~ 0 using the Secant method, within a tolerance
    for the energy 'target', for a quantum oscillator with potential V(x).
    The wavefunction is estimated between x0 and x1.
    Input:
    e1 = lower bracket energy in J
    e2 = upper bracket energy in J
    target = tolerance for energy (float)
    V = potential function
    x-range = [x0, x1] (float)
    Output:
    E = Energy (float) in J
    """
    psi2 = solve(e1, V, x0, x1)[-1] # only want the latest psi value at x1
    while abs(e2 - e1) > target:
        psi1, psi2 = psi2, solve(e2, V, x0, x1)[-1]
        e1, e2 = e2, e2 - psi2 * (e2 - e1) / (psi2 - psi1)
    # Return est current estimate
    return e2

# Crude initial guesses (J): start between 0 and 100 eV
Eg = secant(0, 100*e, e/1000, V_quad, -10*a, 10*a)   # ground state as in example 8.9
E1 = secant(200*e, 300*e, e/1000, V_quad, -10*a, 10*a)  # first excited (use Eg as floor)
E2 = secant(400*e, 700*e, e/1000, V_quad, -10*a, 10*a)  # second excited

print(f'Ground state energy for quadratic E_1 = {Eg/e} eV')
print(f'First excited state energy for quadratic E_2 = {E1/e} eV')
print(f'Second excited state energy for quadratic E_3 = {E2/e} eV')

# Check that our values are evenly spaced (up to our precision):
# Precision chosen was e/1000

print(f'Difference between E_g and E_1 = {abs(Eg/e - E1/e)} eV')
print(f'Difference between E_1 and E_2 = {abs(E2/e - E1/e)} eV')

# Compare to analytic values
omega = np.sqrt(2*V_0/(m*a*a))
print(f'Analytic quadratic E_1 = {1/2 * hbar*omega/e} eV')
print(f'Analytic quadratic E_2 = {3/2 * hbar*omega/e} eV')
print(f'Analytic quadratic E_3 = {5/2 * hbar*omega/e} eV')


# Question 2b): Modifying Example 8.9 for Anharmonic oscillator
# from our previous definitions of functions we simply need to define a new
# potential function

def V_anh(x):
  """
  Returns the value of the anharmonic potential in a harmonic oscillator,
  at position x.
  Input:
  x = Position in space of electron (float)
  Output:
  V = Value of the anharmonic potential (float)
  """
  return V_0 * x**4/a**4

# Calculate the first three energies
Eg_anh = secant(0*e, 100*e, e/1000, V_anh, -10*a, 10*a)   # ground state
E1_anh = secant(500*e, 800*e, e/1000, V_anh, -10*a, 10*a)  # first excited
E2_anh = secant(1000*e, 1200*e, e/1000, V_anh, -10*a, 10*a)  # second excited

print(f'Ground state energy for anharmonic E_1 = {Eg_anh/e} eV')
print(f'First excited state energy for anharmonic E_2 = {E1_anh/e} eV')
print(f'Second excited state energy for anharmonic E_3 = {E2_anh/e} eV')

# Question 2c): Calculate Normalised wavefunctions

# By definition, function 'solve' returns psi(x) for x in the range [x0, x1]
# un-normalised

# Define normalisation function to integrate using Simpson's method

def normalise(E, V, x0, x1):
  """
  Normalise the wavefunction of energy E by integrating the absolute squared
  values of psi outputted by solve on potential V(x), with a lower bound x0
  (Taking advantage of the symmetric, only integrate for 0 to -inf)
  and upper bound x1
  Note: assume that the energy E gives psi with all real values
  Input:
  E = Energy (float), in eV
  psi = output of solve function (float, includes psi(x) for all x from x0 to x1)
  V = potential function
  x-range = [x0, x1] (float)
  Output:
  psi = normalised wavefunction in discrete steps
  """

  psi = solve(E, V, x0, x1)
  # calculate the slice width h
  N = 1000 # this is set for all calculations
  h = (x1 - x0)/N
  # Given wavefunction is symmetric around 0, and assuming x0 and x1 are
  # equally spaced to the left and right of zero, take psi at x0 (negative end)
  # and psi in the middle (we know this is zero)
  area = (psi[0]**2 + psi[N//2]**2)
  # Then add the interior areas in terms of k for odd and even k's separate
  for k in range(1, N//2, 2): # For odd numbers
    area += 4*psi[k]**2
  for k in range(2, N//2, 2): # For even numbers
    area += 2*psi[k]**2
  # Finally multiply by h/3 on the outside of the brackets
  # Return two times the area to account for positive side
  area *= 2*h/3
  return np.asarray(psi)/np.sqrt(area)

# Harmonic Potential
# Normalise
norm_psi_g = normalise(Eg, V_quad, -10*a, 10*a)
norm_psi_1 = normalise(E1, V_quad, -10*a, 10*a)
norm_psi_2 = normalise(E2, V_quad, -10*a, 10*a)

# Analytic solutions
alpha = m*omega/hbar # From website: alpha = mω/ħ

def psi0_an(x):  # n=0
    return (alpha/np.pi)**0.25 * np.exp(-0.5*alpha*x*x)

def psi1_an(x):  # n=1
    return np.sqrt(2*alpha) * x * psi0_an(x)

def psi2_an(x):  # n=2
    return (1/np.sqrt(2)) * (2*alpha*x*x - 1) * psi0_an(x)

x_ran = np.linspace(-10*a, 10*a, 1000)
psi_g_an = psi0_an(x_ran)
psi_1_an = psi1_an(x_ran)
psi_2_an = psi2_an(x_ran)

# Plot both numerical and analytical solutions for all three energies

fig, ax = plt.subplots(3, 1, figsize=(7, 10), sharex=True)

data_pairs = [
    (norm_psi_g, psi_g_an),
    (norm_psi_1, psi_1_an),
    (norm_psi_2, psi_2_an)
    ]

energies = [Eg/e, E1/e, E2/e]

for i in range(len(data_pairs)):
    ax[i].plot(x_ran/a, data_pairs[i][0], lw=1.8, label="Numerical (RK4 + secant)")
    ax[i].plot(x_ran/a, data_pairs[i][1], lw=1.2, ls='--', label="Analytic HO")
    ax[i].set_title(fr"n={i} (E = {round(energies[i],2)} eV)", fontsize=12)
    ax[i].grid(True)


ax[0].legend(loc="upper right")
ax[-1].set_xlabel(r"$x/a$")
ax[1].set_ylabel(r"Normalized $\psi_n(x)$")
plt.tight_layout()
plt.show()

# Anharmonic Potential
# Normalise
norm_psi_g_anh = normalise(Eg_anh, V_anh, -5*a, 5*a)
norm_psi_1_anh = normalise(E1_anh, V_anh, -5*a, 5*a)
norm_psi_2_anh = normalise(E2_anh, V_anh, -5*a, 5*a)

x_ran_anh = np.linspace(-5*a, 5*a, N)

# plot the wavefunctions
plt.plot(x_ran_anh, norm_psi_g_anh, lw=1.8, label= fr"$n=0$ (E = {round(Eg_anh/e,2)} eV)")
plt.plot(x_ran_anh, norm_psi_1_anh, lw = 1.8, label = fr"$n=1$ (E = {round(E1_anh/e,2)} eV)")
plt.plot(x_ran_anh, norm_psi_2_anh, lw = 1.8, label = fr"$n=2$ (E = {round(E2_anh/e,2)} eV)")
plt.legend()
plt.xlabel(r"Position $x/a$")
plt.ylabel(r"Normalized $\psi_n(x)$")
plt.title("Anharmonic Oscillator")
plt.grid()
plt.show()
