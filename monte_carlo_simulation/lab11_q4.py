# Question 4
"""
Author: Erin Fung anf Valerie MacIsaac

Explanatory Notes: This exercise is from Newman 10.9. It explores the Ising
model.
"""

# Import modules
import numpy as np
from random import random, randrange
import matplotlib.pyplot as plt

# define constants
kB = 1.0
T = 1.0
J = 1.0
N = 100000

def energyfunction(J_, dipoles):
    """ function to calculate energy """
    # sum over all row neighbors first (horizontal)
    E = -J_ * np.sum(dipoles[:, :-1] * dipoles[:, 1:])
    # then sum over all column neighbors (vertical)
    E += -J_ * np.sum(dipoles[:-1, :] * dipoles[1:, :])
    return E


def acceptance(E_new, E):
    """ Function for acceptance probability"""
    p = np.exp(-(E_new - E)/(kB*T))
    if E_new - E <= 0:
        return True
    elif (E_new-E) > 0 and (p > random()):
        return True
    else:
        return False

def metropolis(E, dipoles):
    """
    Flip a spin state randomly and decide whether to accept the flip, given
    current energy E and dipole state dipoles
    """
    nrows, ncols = dipoles.shape
    picked_i, picked_j = randrange(nrows), randrange(ncols) # choose a victim
    dipoles[picked_i, picked_j] *= -1  # propose to flip the victim
    Enew = energyfunction(J, dipoles)  # compute Energy of proposed new state
    M = np.sum(dipoles) # compute magnetisation of proposed new state
    # Decide whether to accept it
    accept = acceptance(Enew, E)

    if not accept:
        dipoles[picked_i, picked_j] *= -1 # flip it back
        M = np.sum(dipoles) # recalculate original magnetisation
    else:
        E = Enew  # accept the energy

    return dipoles, E, M

# generate array of dipoles and initialize diagnostic quantities

num_dipoles = 20 # number of spins in each direction
dipoles = np.ones((num_dipoles * num_dipoles), int)
dipoles[: dipoles.size // 2] = -1 # for half of the ones set it to -1
dipoles = dipoles.reshape((num_dipoles, num_dipoles)) # reshape it so it's the correct 2x2 dimensions
np.random.shuffle(dipoles)  # randomly shuffle them

energy = []  # empty list; to add to it, use energy.append(value)
magnet = []  # empty list; to add to it, use magnet.append(value)

E = energyfunction(J, dipoles)
energy.append(E)
magnet.append(np.sum(dipoles))


for i in range(N):
    dipoles, E, M = metropolis(E, dipoles)
    energy.append(E)
    magnet.append(M)

    # animate every 1000 steps
    if i % 1000 == 0:
        plt.imshow(dipoles, cmap='plasma', vmin=-1, vmax=1)
        plt.colorbar(label='Spin')
        plt.title(f'Spin configuration at step {i}')
        plt.xlabel('Column index')
        plt.ylabel('Row index')
        plt.tight_layout()
        plt.show()

# plot energy, magnetisation as function of steps

plt.plot(range(N+1), energy, '-')
plt.xlabel('Steps (n)')
plt.ylabel('Energy (J)')
plt.title('Energy per Step')
plt.grid()
plt.show()

plt.plot(range(N+1), magnet, '-')
plt.xlabel('Steps (n)')
plt.ylabel('Magnetisation')
plt.title('Magnetisation per Step')
plt.grid()
plt.show()
