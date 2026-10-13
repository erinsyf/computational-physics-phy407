# Question 3:
"""
Author: Valerie MacIsaac and Erin Fung

Explanatory Notes: This question compares the mean value and importance sampling
Monte Carlo integral methods.
"""

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
plt.rcParams['figure.dpi'] = 400 # set higher resolution figures


# =========================== Part a) ================================

# Write a function to do both methods so we can repeat for 100 times per method

def both_methods():
    """
    Evaluate the integral from equation (8) using mean value method and
    importance sampling. Here N = 10000.
    Output: The integral evaluated by mean value method and importance sampling
    I_mv and I_is
    """
    N = 10000 # sample points
    b = 1.0 # upper bound of integral
    a = 0.0 # lower bound of integral

    # 1. mean value method

    # Calculate the sum
    x_i = a + (b - a) * np.random.random(N) # choose N random values of x_i between b and a
    I_mv_sum = np.sum((x_i)**(-0.5)/(1+np.exp(x_i)))

    # final estimation using mean value
    I_mv = (b-a)/N * I_mv_sum

    # 2. Importance Sampling

    # Calculate the sum

    # sum of f(x_i)/w(x_i)*integral of w(x) from a to b
    z = np.random.random(N) # between 0 and 1
    x_j = (z)**2 # random x_j falling under probability distribution p(x)
    I_is_sum = np.sum(2/(1+np.exp(x_j)))

    I_is = 1/N * I_is_sum

    return I_mv, I_is

# now repeat the integrals 100 times, and collect the integrals
I_mv_repeated = []
I_is_repeated = []

for _ in range(100):
    I_mv, I_is = both_methods()
    I_mv_repeated.append(I_mv)
    I_is_repeated.append(I_is)

# plot histograms for the resulting integral values for 10 bins

plt.hist(I_mv_repeated, 10, range=[0.8, 0.88])
plt.xlabel('Integral Value')
plt.ylabel('Frequency')
plt.title('Integral distribution for Mean Value Method')
plt.grid()
plt.show()

plt.hist(I_is_repeated, 10, range=[0.8, 0.88])
plt.xlabel('Integral Value')
plt.ylabel('Frequency')
plt.title('Integral distribution for Importance Sampling')
plt.grid()
plt.show()

# ======================== Part b) ==============================================

# Consider the nonsingular integrand this time,


def both_methods_1():
    """
    Evaluate the integral from equation (10) using mean value method and
    importance sampling. Here N = 10000.
    Output: The integral evaluated by mean value method and importance sampling
    I_mv and I_is
    """
    N = 10000 # sample points
    b = 10.0 # upper bound of integral
    a = 0.0 # lower bound of integral

    # 1. mean value method


    # Calculate the sum
    x_i = a + (b - a) * np.random.random(N) # choose random value of x_i between b and a
    I_mv_sum = np.sum(np.exp(-2 * np.abs(x_i - 5)))

    # Final mean value estimation
    I_mv = (b-a)/N * I_mv_sum

    # 2. Importance Sampling

    # Calculate the sum
    # Note here Sum of f(x)/w(x) * integral w(x) = sum of f(x)/p(x)
    x_j = np.random.normal(5, 1, N)
    p_x = (1/np.sqrt(2*np.pi))*np.exp(-0.5*(x_j-5)**2)
    f_x = np.exp(-2*np.abs(x_j-5))

    # pick out the correct range

    p_x_safe = p_x[(x_j >= 0.0) & (x_j <= 10.0)]
    f_x_safe = f_x[(x_j >= 0.0) & (x_j <= 10.0)]

    I_is_sum = np.sum(f_x_safe/p_x_safe)

    # final value is mean
    I_is = 1/N * I_is_sum

    return I_mv, I_is

# now repeat the integrals 100 times, and collect the integrals
I_mv_repeated_1 = []
I_is_repeated_1 = []

for _ in range(100):
    I_mv_1, I_is_1 = both_methods_1()
    I_mv_repeated_1.append(I_mv_1)
    I_is_repeated_1.append(I_is_1)

# plot histograms for the resulting integral values for 10 bins

plt.hist(I_mv_repeated_1, 10)
plt.xlabel('Integral Value')
plt.ylabel('Frequency')
plt.title('Integral distribution for Mean Value Method')
plt.grid()
plt.show()

plt.hist(I_is_repeated_1, 10)
plt.xlabel('Integral Value')
plt.ylabel('Frequency')
plt.title('Integral distribution for Importance Sampling')
plt.grid()
plt.show()
