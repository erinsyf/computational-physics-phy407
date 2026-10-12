# Question 2
'''
Author: Valerie MacIsaac and Erin Fung

Explanatory Notes: This question solves the Schrodinger equation using the
Crank-Nicolson method. The behavior is explored for a square well, a double
square well, and a harmonic oscillator.
'''

# Import Libraries
from scipy.constants import m_e, hbar
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import rc



# Constants
L = 1e-8 # meters
sigma = L/25
kappa = 500/L
P = 1024
tau = 1e-18 # time step in seconds
delt = L/P # step size in spatial domain dx

# discretised x values consistent for each part
x = np.linspace(-L/2, L/2, P, float)

# Create Recurring functions used to operate the algorithm

def create_matrix(delt, V, P):
  """
  Creates a square matrix Hamiltonian of size 'P' (for the interior points, so
  the matrix size is P-2), with a certain step size 'delt'
  (in the spatial domain x), for a given potential 'V'
  Input:
  step_size = step size in x (float, in meters)
  V = Potential function
  size = size of matrix (integer)
  Output:
  size x size Hamiltonian array
  """
  P_int = P - 2
  A_val = - hbar**2 / (2*m_e*delt**2)

  # interior x positions (exclude walls)
  p = np.arange(1, P-1)
  B = V(p*delt - L/2) - 2*A_val
  A  = A_val * np.ones(P_int - 1)
  H = (
        np.diag(B) +      # Main diagonal
        np.diag(A, k=1) +  # Upper diagonal
        np.diag(A, k=-1)   # Lower diagonal
      )

  return H

def run_diagnostics(H, wave, delt):
  """
  Compute the expectation value of energy <H>, position <x> and normalisation
  at certain time t.
  Input:
  H = Hamiltonian
  wave = Wavefunction at certain time t
  Output:
  E = Expectation value of energy
  X = Expectation value of position
  norm = normalisation
  """
  # For expectation values, just use discretised Hamiltonian and
  # arrays of wavefunctions as vectors and matrices
  psi_int = wave[1:-1] # get interior of wavefunction
  E   = np.real(integrate.simpson((psi_int.conj()*(np.dot(H, psi_int))), dx=delt)) # <psi|H|psi> = <H> = E
  X   = integrate.simpson(np.conj(psi_int) * x[1:-1] * psi_int, dx=delt) # <x> = <psi|x|psi>
  squared = (np.abs(psi_int)**2)
  norm= integrate.simpson(squared, dx=delt) # Integral of |psi|^2
  return E, X, norm

def create_psi(x_0):
  """
  Create normalised initial wavefunction psi(x, t = 0) according to equation 21
  Input:
  x_0 = constant in initial condition
  Output:
  psi = Wavefunction at initial time t=0
  """
  exponential =  np.exp(-(x-x_0)**2/(4*sigma**2) + 1j*kappa*x)
  # calculate normalisation: where the imaginary part gets cancelled in the
  # integral of |psi(x)|^2
  norm = integrate.simpson(np.exp(-(x-x_0)**2/(2*sigma**2)), dx=delt)
  psi = exponential / np.sqrt(norm)
  # enfore boundary conditions
  psi[0]=psi[-1]=0
  return psi

def crank_nicolson(H, psi, tau, N):
  """
  Run the Crank-Nicolson algorithm for a given time step tau, with Hamiltonian
  H, and initial wavefunction psi, for total timesteps N
  Output:
  A 2x2 array with time in the first index, and psi for all x in the second
  """

  psi_int = psi[1:-1].copy() # interior points length P-2
  P = len(psi) # total P
  P_int = len(psi_int)
  I = np.identity(P_int, dtype=complex) # identity size of interior points
  L = I + 1j*tau/(2*hbar)*H
  R = I - 1j*tau/(2*hbar)*H

  psi_series = np.zeros((N, P), complex)
  for n in range(N):
    V = np.dot(R, psi_int)
    psi_int = np.linalg.solve(L, V)
    psi_full = np.zeros(P, complex) # create full psi length P
    psi_full[1:-1] = psi_int # Add the interior back into the full psi
    psi_series[n,:] = psi_full # Append into collective series
  return psi_series

def save_animation(psi, time_series, title):
    """
    Make and save the animation for psi with title, across times in time_series
    Adapted from N.G's code
    Output:
    Nothing, saves animation
    """
    font = {'size': 14}  # adjust fonts
    rc('font', **font)

    fig, ax = plt.subplots()
    ln, = plt.plot([], [])

    X, T = np.meshgrid(x, time_series)

    data = np.real(psi)

    ymax = 1.05 * np.max(np.abs(data))
    ymin = -ymax

    def init():
        ax.set_xlim(-L/2, L/2)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(r'$x$ (m)')
        ax.set_ylabel(r'$\mathbb{{R}}(\psi(x,t))$ $(m^{-1})$')
        ax.grid(True)

        return ln,


    def to_be_animated(idx):
        tval = time_series[idx]
        ax.set_title(r'$\mathbb{{R}}(\psi(x,t))$ at $t = {:.2e}$ s'.format(tval))
        plt.tight_layout()
        ln.set_data(x, data[idx, :])
        return ln,


    ani = FuncAnimation(fig,  # figure identifyer object
                        to_be_animated,  # the function to animate in time
                        frames=range(0, len(time_series), 4),  # plotting every four frames
                        init_func=init,
                        interval=100,  # [s]  time between each frame
                        blit=True)  # blit speeds up the execution of script

    ani.save(title, writer='ffmpeg', dpi=100)


# ======================= Infinite Square Well ============================

x_0_square = L/5 # x_0 for square well
N_square = 3000 # number of total time steps
T_square = N_square*tau # end time

# Define potential
def V_square_well(x):
  """
  Potential of an infinite square well at positions x
  """
  return np.zeros_like(x)

# Define the matrix H_D
H_D_square_well = create_matrix(delt, V_square_well, P)

# Define initial wavefunction
psi_square_well = create_psi(x_0_square)

# Calculate psi using Crank-Nicolson
psi_square_well_series = crank_nicolson(H_D_square_well, psi_square_well, tau, N_square)

# Check diagnostics
energies_square = np.zeros(N_square, complex) # store energies
positions_square = np.zeros(N_square, complex) # store position
norms_square = np.zeros(N_square, complex) # store normalisations

for i in range(N_square):
  E, X, norm = run_diagnostics(H_D_square_well, psi_square_well_series[i, :], delt)
  energies_square[i] = E
  positions_square[i] = X
  norms_square[i] = norm

# Plot the diagnostic values
time_series_square = np.linspace(0, T_square, N_square)
fig, axes = plt.subplots(3,1, figsize=(7, 10), sharex=True)
axE, axX, axN = axes
for ax in axes:
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))

# Energy
axE.plot(time_series_square, energies_square)
axE.set_xlabel('Time (s)')
axE.set_ylabel('Energy (J)')
axE.set_ylim(energies_square[0]*(0.8), energies_square[0]*1.2)
axE.set_title('Expectation value of Energy')
axE.grid(True)

# Position
axX.plot(time_series_square, positions_square)
axX.set_xlabel('Time (s)')
axX.set_ylabel('Position (m)')
axX.set_title('Expectation value of Position')
axX.grid(True)

# Normalisation
axN.plot(time_series_square, norms_square)
axN.set_xlabel('Time (s)')
axN.set_ylabel('Normalisation')
axN.set_ylim(0, 2)
axN.set_title('Normalisation over Time')
axN.grid(True)

fig.tight_layout()
plt.show()

# Behaviour of wavefunction

save_animation(psi_square_well_series, time_series_square, 'square_well_animation.mp4')

# ========================= Harmonic Potential ==============================

omega = 3e15 # radians per second
x_0_harmonic = L/5
N_harmonic = 4000
T_harmonic = N_harmonic*tau

# Define potential
def V_harmonic(x):
  """
  Potential of an harmonic oscillators for array x
  """
  return 0.5 * m_e * (omega**2) * x**2

# Define the matrix H_D
H_D_harmonic = create_matrix(delt, V_harmonic, P)

# Define initial wavefunction
psi_harmonic = create_psi(x_0_harmonic)

# Calculate psi using Crank-Nicolson
psi_harmonic_series = crank_nicolson(H_D_harmonic, psi_harmonic, tau, N_harmonic)

# Check diagnostics
energies_harmonic = np.zeros(N_harmonic, complex) # store energies
positions_harmonic = np.zeros(N_harmonic, complex) # store position
norms_harmonic = np.zeros(N_harmonic, complex) # store normalisations

for i in range(N_harmonic):
  E, X, norm = run_diagnostics(H_D_harmonic, psi_harmonic_series[i, :], delt)
  energies_harmonic[i] = E
  positions_harmonic[i] = X
  norms_harmonic[i] = norm

# Plot the diagnostic values
time_series_harmonic = np.linspace(0, T_harmonic, N_harmonic)
fig, axes = plt.subplots(3,1, figsize=(7, 10), sharex=True)
axE, axX, axN = axes
for ax in axes:
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))

# Energy
axE.plot(time_series_harmonic, energies_harmonic)
axE.set_xlabel('Time (s)')
axE.set_ylabel('Energy (J)')
axE.set_ylim(energies_harmonic[0]*(0.8), energies_harmonic[0]*1.2)
axE.set_title('Expectation value of Energy')
axE.grid(True)

# Position
axX.plot(time_series_harmonic, positions_harmonic)
axX.set_xlabel('Time (s)')
axX.set_ylabel('Position (m)')
axX.set_title('Expectation value of Position')
axX.grid(True)

# Normalisation
axN.plot(time_series_harmonic, norms_harmonic)
axN.set_xlabel('Time (s)')
axN.set_ylabel('Normalisation')
axN.set_ylim(0, 2)
axN.set_title('Normalisation over Time')
axN.grid(True)

fig.tight_layout()
plt.show()

# Behaviour of wavefunction

save_animation(psi_harmonic_series, time_series_harmonic, 'harmonic_animation.mp4')

# ============================= Double Well ==================================

V_0 = 6e-17 # J
x_0_double = L/3
x_1 = L/4
N_double = 6000
T_double = N_double*tau

# Define potential
def V_double_well(x):
    """
    Double well potential for an array x
    """
    return V_0 *(x**2/x_1**2 - 1 )**2

# Define the matrix H_D
H_D_double = create_matrix(delt, V_double_well, P)

# Define initial wavefunction
psi_double = create_psi(x_0_double)

# Calculate psi using Crank-Nicolson
psi_double_series = crank_nicolson(H_D_double, psi_double, tau, N_double)

# Check diagnostics
energies_double = np.zeros(N_double, complex) # store energies
positions_double = np.zeros(N_double, complex) # store position
norms_double = np.zeros(N_double, complex) # store normalisations

for i in range(N_double):
  E, X, norm = run_diagnostics(H_D_double, psi_double_series[i, :], delt)
  energies_double[i] = E
  positions_double[i] = X
  norms_double[i] = norm

# Plot the diagnostic values
time_series_double = np.linspace(0, T_double, N_double)
fig, axes = plt.subplots(3,1, figsize=(7, 10), sharex=True)
axE, axX, axN = axes
for ax in axes:
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))

# Energy
axE.plot(time_series_double, energies_double)
axE.set_xlabel('Time (s)')
axE.set_ylabel('Energy (J)')
axE.set_ylim(energies_double[0]*(0.8), energies_double[0]*1.2)
axE.set_title('Expectation value of Energy')
axE.grid(True)

# Position
axX.plot(time_series_double, positions_double)
axX.set_xlabel('Time (s)')
axX.set_ylabel('Position (m)')
axX.set_title('Expectation value of Position')
axX.grid(True)

# Normalisation
axN.plot(time_series_double, norms_double)
axN.set_xlabel('Time (s)')
axN.set_ylabel('Normalisation')
axN.set_ylim(0, 2)
axN.set_title('Normalisation over Time')
axN.grid(True)

fig.tight_layout()
plt.show()
