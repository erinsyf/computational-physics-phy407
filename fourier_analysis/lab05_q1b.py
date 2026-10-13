# Question 1b
"""
Author: Erin Fung and Valerie MacIsaac

Explanatory Notes: Code for Exercise 7.1 and 7.8 from Newman.

b) The second portion of the code focuses on exercise 7.8. This exercise uses
fourier transforms to explore the intensity pattern of a diffraction grating.
By padding the spaces to the left and right of the aperture width, we can apply
a fourier transform that generates results very similar to another integration
method (Newman exercise 5.19). For this question, we began by defining a
transmission function, and then finding the fourier coefficients for a padded
array of those values. We then shifted the fourier transform and plotted the
theoretical instensity spectrum and the fourier intensity spectrum.
"""

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
from pylab import imshow
import scipy.integrate as integrate

plt.rcParams['figure.dpi'] = 300 # set higher resolution figures

# Defining constants used in diffraction intensity equation
w = 200e-6 # meters
W = 10*w # aperture width in meters
wavelength = 500e-9 # meters
f = 1 # focal length in meters
L = 10e-2 # width of screen in meters

def y_n(u):
  """
  Returns the value of the square root of the transmission function at u
  Input:
  u = Float
  Output:
  Value of the transmission function at u
  """
  alpha = np.pi/(20e-6) # in m^-1
  return np.sin(alpha*u)**2

# Create an array of values to apply the Fourier transform to

dx = wavelength * f / w # spacing on physical screen between FFT samples
N = max(int(L/dx), int(10*W/w)) # number of points to evaluate in the integral
du = w / N # Distance between points evaluated across aperture
u = np.linspace(-w/2, w/2, N) # Values across aperture u from -W/2 to W/2
x = np.linspace(-L/2, L/2, N) # values across screen x from -5 cm to 5 cm

E_k = np.array([np.sqrt(y_n(i)) for i in u])

# Increase aperture width by 10x
pad_factor = 10
N_new = pad_factor * N

# Zero-pad symmetrically around the points we already have
pad_before = (N_new - N) // 2 # number of zeros to add in front of array
pad_after = N_new - N - pad_before # number of zeros left to add in back of array

E_k_padded = np.pad(E_k, (pad_before, pad_after), mode='constant', constant_values=0)

# Define new u-grid spanning W = 10 * w
u_new = np.linspace(-5*W, 5*W, N_new)  # assuming original u was centered on 0

c_k = fft.fft(E_k_padded)
freq = fft.fftfreq(N_new, d = du)
# ---- shift to center zero frequency ----
c_k_shifted = np.fft.fftshift(c_k)
freq_shifted = np.fft.fftshift(freq)

# Plot the frequency spectrum of the transmission function

plt.plot(freq_shifted, np.abs(c_k_shifted))
plt.title('Frequency Spectrum of Transmission Function')
plt.xlabel('Frequency')
plt.ylabel('Magnitude $|c_k|$')
plt.grid()
plt.show()

# Create Intensity values
I_k = (W**2 / N_new**2) * np.abs(c_k_shifted)**2

# Map frequencies to screen position x for the new aperture size
k_shifted = np.arange(-N_new//2, N_new//2) # shift back so zero lies in the middle
x_screen = wavelength * f/W * k_shifted

# Plot intensity vs position
plt.plot(x_screen, I_k, label = 'Fourier')
plt.xlim(-0.05, 0.05) # limit it to the screen size

# Expected result by Simpson's integration of 5.19
I_expected = []
for p in x:
  # Because the integral inside I is imaginary, integrate real and imaginary
  # parts separately
  I_real = integrate.simpson(np.sqrt(y_n(u))*np.cos(2*np.pi * p * u /(wavelength *f)), u)
  I_im = integrate.simpson(np.sqrt(y_n(u))*np.sin(2*np.pi * p * u /(wavelength *f)), u)
  I_expected.append(I_real**2 + I_im**2)
# Plot theoretical/expected function
plt.plot(x, I_expected, '--', alpha = 0.7, label = 'Integrated')
plt.title("Diffraction Intensity vs Screen Position")
plt.xlabel("Position on screen x (meters)")
plt.ylabel("I(x) $(W/m^2)$")
plt.grid()
plt.legend()
plt.show()
