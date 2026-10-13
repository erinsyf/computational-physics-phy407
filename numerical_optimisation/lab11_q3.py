# Question 3a)
"""
Author: Erin Fung anf Valerie MacIsaac

Explanatory Notes: This code explores an example of a simulated annealing
optimization problem, specifically the travelling salesman problem.
"""

# Import Libraries
from math import sqrt,exp
from numpy import empty, linspace
from random import random,randrange
from random import seed
import matplotlib.pyplot as plt


def run_salesman(ns2, tau):
    """

    Parameters
    ----------
    ns2 : Integer, the second seed passed into optimisation before while loop
    The first seed is fixed.
    tau : cooling rate
    Returns 2 d array for r[N, 2]
    -------

    """

    N = 25
    R = 0.03
    Tmax = 1.0
    Tmin = 1e-3

    # Function to calculate the magnitude of a vector
    def mag(x):
        return sqrt(x[0]**2+x[1]**2)

    # Function to calculate the total length of the tour
    def distance():
        s = 0.0
        for i in range(N):
            s += mag(r[i+1]-r[i])
        return s

    # Choose N city locations and calculate the length of the route and
    # the initial value of the distance
    r = empty([N+1,2],float)

    # Seed random locations (this changes where the cities are placed)
    ns = 6
    seed(ns)

    for i in range(N):
        r[i,0] = random()
        r[i,1] = random()
    r[N] = r[0]
    D = distance()

    # Main loop
    t = 0
    T = Tmax

    # Second seed to control optimisation path
    seed(ns2)
    while T>Tmin:
        # Cooling
        t += 1
        T = Tmax*exp(-t/tau)

        # Choose two cities to swap and make sure they are distinct
        i,j = randrange(1,N),randrange(1,N)
        while i==j:
            i,j = randrange(1,N),randrange(1,N)

        # Swap them and calculate the change in distance
        oldD = D
        r[i,0],r[j,0] = r[j,0],r[i,0]
        r[i,1],r[j,1] = r[j,1],r[i,1]
        D = distance()
        deltaD = D - oldD

        # If the move is rejected, swap them back again
        if random()>exp(-deltaD/T):
            r[i,0],r[j,0] = r[j,0],r[i,0]
            r[i,1],r[j,1] = r[j,1],r[i,1]
            D = oldD
    # return the city  and final distance D
    return r, D

# Three values of tau, original  tau = 1e4

taus = [1e3, 1e4, 1e5]

# different seeds for the while loop
ns_list = [1, 10, 20, 30, 40, 50]
total_distances = empty([len(taus), len(ns_list)], float)

for i, tau in enumerate(taus):

    for j, ns2 in enumerate(ns_list):
        r, D = run_salesman(ns2, tau)
        total_distances[i,j] = D
        plt.plot(r[:, 0], r[:, 1], 'o-', alpha = 0.6, label = f'ns2 = {ns2}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(rf'Travelling Salesman Path for $\tau = {tau}$')
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.grid()
    plt.show()

# Plot total distance as a function of seed for each tau to compare them

for i, tau in enumerate(taus):
    plt.plot(ns_list, total_distances[i, :], 'o-', label = rf'$\tau = {tau}$')
plt.xlabel('Seed Value ns2')
plt.ylabel('Total Distance D')
plt.title(r'Total Distance D vs seed ns for each $\tau$')
plt.grid()
plt.legend()
plt.show()


# Question 3b)
"""
Author: Erin Fung and Valerie MacIsaac

Explanatory Notes: This code is a second example of a simulated annealing
problem. Adapted from Newman, exercise 10.10, this problem explores finding
global minimums of functions using simulated annealing.
"""
# Import Libraries
from math import sqrt,exp
import numpy as np
from random import random
import matplotlib.pyplot as plt


# function to minimise for part i
def f_i(list_xy):
    x = list_xy[0]
    y = list_xy[1]
    return x**2 - np.cos(4*np.pi*x) +(y-1)**2

# function to minimise for part ii
def f_ii(list_xy):
    x = list_xy[0]
    y = list_xy[1]
    return np.cos(x)+ np.cos(sqrt(2)*x)+ np.cos(sqrt(3)*x)+(y-1)**2

def find_minimum(func, initial_xy, Tmax, Tmin, tau, part_ii):
    """
    Finds minimum of function func, with initial value of (x,y) given.
    if part_ii is True then it activates
    condition on (x,y) to bound it in 0 < x < 50, −20 < y < 20.
    Inputs:
        func = function to optimise
        initial_xy = list of initial (x,y)
        T_max = Starting temperature
        T_min = minimum temperature to end run
        tau = cooling rate
        part_ii = Boolean to limit x and y
    Outputs:
        Final (x,y), list of x and y values throughout the run, and final f(x,y)
    """

    # Initial (x, y)
    x_y = np.array(initial_xy, float)

    # claculate initial value of f(x,y)
    f_xy = func(x_y)

    # Main loop
    t = 0
    T = Tmax

    # Track the changes in (x,y)

    x_list = [x_y[0]]
    y_list = [x_y[1]]

    while T>Tmin:

        # Cooling
        t += 1
        T = Tmax*exp(-t/tau)

        #  calculate new (x,y) point, randomly drawn from Gaussian

        # first in polar coordinates
        z = random()

        # random radius between 0 and 1 normally distributed (equation 10.17)
        r_random = sqrt(-2*np.log(1-z))

        # Random angle between 0 and 2pi
        theta_random = random() * 2 *np.pi

        # Calculate changes in x and y for one monte-carlo move deltax and deltay
        delta_x = r_random*np.cos(theta_random)
        delta_y = r_random*np.sin(theta_random)

        # Calculate the new x and y
        x_y[0] += delta_x
        x_y[1] += delta_y

        # If it's part ii then bound (x,y), keep generating random (x,y) until
        # in the correct range
        if part_ii:
            while (abs(x_y[1]) >= 20) or (x_y[0] <= 0) or (x_y[0] >= 50):
                # Get back the original x_y, and try again
                x_y[0] -= delta_x
                x_y[1] -= delta_y
                z = random()
                r_random = sqrt(-2*np.log(1-z))
                theta_random = random() * 2 *np.pi
                delta_x = r_random*np.cos(theta_random)
                delta_y = r_random*np.sin(theta_random)

                # Calculate the new x and y
                x_y[0] += delta_x
                x_y[1] += delta_y

        # Swap them and calculate the change in f(x,y)
        oldf = f_xy
        f_xy = func(x_y)
        deltaf = f_xy - oldf

        # If the move is rejected, subtract the changes in x and y back to original
        if random()>exp(-deltaf/T):
            x_y[0] -= delta_x
            x_y[1] -= delta_y
            f_xy = oldf

        # append the current iterations x_y value
        x_list.append(x_y[0])
        y_list.append(x_y[1])

    # return the final value of (x,y), f(x,y), and list of (x,y) with respect to time
    return x_y, f_xy, x_list, y_list


x_y_i, f_xy_i, x_list_i, y_list_i = find_minimum(f_i, [2.0, 2.0],10.0, 1e-5, 1e4, False)
print("Final (x,y) for part i  = ", x_y_i)
print("f(final) for part i  = ", f_xy_i)

# Plot points (x,y) as function of time during run

plt.plot(x_list_i, y_list_i, 'o', ms = 2.5)
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.title('Points (x,y) as Function of Time During Run')
plt.show()

x_y_ii, f_xy_ii, x_list_ii, y_list_ii = find_minimum(f_ii, [20.0, 5.0],10.0, 1e-3, 1e4, True)
print("Final (x,y) for part ii = ", x_y_ii)
print("f(final) for part ii = ", f_xy_ii)

# Plot points (x,y) as function of time during run

plt.plot(x_list_ii, y_list_ii, 'o', ms = 2.5)
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.title('Points (x,y) as Function of Time During Run')
plt.show()
