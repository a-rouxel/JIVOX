import imageio
import matplotlib.pyplot as plt
import numpy as np
from jax import jit
import jax.numpy as jnp
from jax import device_put
from jax.scipy.signal import convolve2d
import time
import os


def crop_padded_image(padded_image, kernel_size):
    """
    Crops a padded image back to its original size after convolution.

    :param padded_image: 2D NumPy array representing the padded image
    :param kernel_size: Size of the convolution kernel used
    :return: Cropped image
    """
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    crop_width = (kernel_size - 1) // 2
    cropped_image = padded_image[crop_width:-crop_width, crop_width:-crop_width]

    return cropped_image

def pad_image(image, kernel_size, mode='constant', constant_values=0):
    """
    Pads an image given the kernel size.

    :param image: 2D NumPy array representing the image
    :param kernel_size: Size of the convolution kernel
    :param mode: Padding mode ('constant', 'edge', 'reflect', etc.)
    :param constant_values: Values for padding in 'constant' mode
    :return: Padded image
    """
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    pad_width = (kernel_size - 1) // 2
    return np.pad(image, pad_width, mode=mode, constant_values=constant_values)


# Function Definitions
def normalize_and_convert(array):
    # Normalize the array to 0-1 range and then scale to 0-255

    array_normalized = (array - array.min()) / (array.max() - array.min())


    array_normalized = (255 * array_normalized).astype(np.uint8)
    return array_normalized

def load_png_mask(dir_path, file_name):
    img = imageio.v2.imread(dir_path + file_name)
    return img

def mask_to_binary(mask):
    mask_ = mask[:,:,3] > 0
    return np.abs(mask_ - 1)

def xy_to_rtheta_2D(x, y):
    x_grid, y_grid = np.meshgrid(x, y)
    r = np.sqrt(x_grid**2 + y_grid**2)
    theta = np.arctan2(y_grid, x_grid)
    return r, theta

def make_the_kernel_odd(noyau):
    # Check if the number of rows is even
    if noyau.shape[0] % 2 == 0:
        # Add a row of zeros
        zero_row = np.zeros((1, noyau.shape[1]))
        noyau = np.vstack([noyau, zero_row])

    # Check if the number of columns is even
    if noyau.shape[1] % 2 == 0:
        # Add a column of zeros
        zero_col = np.zeros((noyau.shape[0], 1))
        noyau = np.hstack([noyau, zero_col])

    return noyau

@jit
def apply_convolution(Mesa_shape, noyau):
    Mesa_shape_jax = device_put(Mesa_shape)
    noyau_jax = device_put(noyau)
    Ox_shape_jax = convolve2d(Mesa_shape_jax, noyau_jax, mode='same')
    rg_tmp = Ox_shape_jax != 0
    return jnp.where(rg_tmp, 1, Ox_shape_jax)