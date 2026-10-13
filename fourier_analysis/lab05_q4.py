# Question 4
"""
Author: Erin Fung and Valerie MacIsaac

Explanatory Notes: This question aims to unblur an image of a house. It begins by
printing the blurred image. An array containing a grid of samples from the
Gaussian spread is then created using a defined Gaussian Spread function.
Both the original data and the spread function data are then Fourier transformed,
divided, and an inverse transform is performed. The final result is plotted.
"""

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft2, irfft2, fftshift
from pylab import imshow, gray, show
plt.rcParams['figure.dpi'] = 300 # set higher resolution figures

# Import data blur.txt
blur_data = np.loadtxt('blur.txt')

# a) Plot the blurred image on grayscale
print(blur_data.shape) # check the shape of the image
K = blur_data.shape[0] # width of image
L = blur_data.shape[1] # height of image

imshow(blur_data, aspect = 'equal')
gray()
show()

# b) Create an array containing grid of samples drawn from the Gaussian spread

def gauss_f(x, y):
  """Returns the value of the Gaussian spread at point (x, y) with standard
  deviation of 25.
  Input:
  (x,y) = Tuple of floats describing the position
  Output:
  Value of the Gaussian spread at point (x, y)
  """
  sigma = 25 # for this question
  return np.exp(-(x**2 + y**2)/(2*sigma**2))

# Initialise an empty array of the same size as the image
spread = np.empty([K, L])

# Fill out the list with Gaussian spread values
for i in range(K):
  for j in range(L):
    # wraps indices so they repeat periodically
    x = (j + K//2) % K - K//2 # if in the second half, it goes back to zero from the negative end
    y = (i + L//2) % L - L//2
    spread[i, j] = gauss_f(x, y)

# Plot the Point Spread Function to check it's as we expect
imshow(spread, aspect = 'equal')
gray()
show()

# c) Combine programs to Fourier transform both, then perform an inverse transform
# to unblur the photo, displaying the final result

blurred_fft = rfft2(blur_data)
point_spread_fft = rfft2(spread)

# Divide them using equation from instructions to get a_k unblurred Fourier

unblurred_ft = np.empty([blurred_fft.shape[0], blurred_fft.shape[1]], complex) # make sure the array is complex valued

err = 10 ** -3 # Define error to prevent division by zero
for i in range(blurred_fft.shape[0]):
  for j in range(blurred_fft.shape[1]):
        if abs(point_spread_fft[i, j]) < err:
            unblurred_ft[i, j] = blurred_fft[i, j]
        else:
            unblurred_ft[i, j] = blurred_fft[i, j] / (point_spread_fft[i, j])

# inverse Fourier transform to get the original image
unblurred_data = irfft2(unblurred_ft)
imshow(unblurred_data, aspect = 'equal')
gray()
show()
