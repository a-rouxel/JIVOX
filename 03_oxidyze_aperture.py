"""
Oxidation Process Simulation for VCSELs
=======================================

This script simulates the oxidation process for Vertical-Cavity Surface-Emitting Lasers (VCSELs).
It uses a set of predefined parameters and input masks to model the oxidation effects on VCSELs
over a specified number of iterations. The script allows for the examination of different aluminum
concentrations and their impact on the oxidation process.

Key Parameters:
----------------
- WINDOW_WIDTH: Width of the window in micrometers (µm).
- L_OX_ref: Reference length for oxidation.
- ANISO: Anisotropy constant.
- nb_of_vcsel_to_oxydize: Number of VCSELs to be oxidized.
- nb_of_oxyd_iter: Number of oxidation iterations.
- input_dir: Directory containing the input dataset.
- speed_depending_on_c: Dictionary mapping aluminum concentrations to their respective oxidation speeds.
- names_oxydize_aperture: List of names for different apertures to be oxidized.

Functionality:
--------------
1. The script iterates over a specified number of VCSELs for oxidation.
2. For each VCSEL, it loads an initial mask and converts it to a binary format.
3. It then calculates an oxidation kernel based on the specified aluminum concentration and anisotropy.
4. The script applies this kernel iteratively to simulate the oxidation process.
5. The resulting images for each iteration are saved in a specified output directory.

Usage:
------
To run the script, ensure that all dependencies are installed and the required input files are present in the 'input_dir'.
Adjust the parameters as necessary for your specific simulation requirements.

Author: A. Rouxel
Date: 22/01/2024
"""


import imageio
import matplotlib.pyplot as plt
import yaml
from dataset_generation.oxidation.functions_oxidation import *  # Importing required modules and functions

# Load the configuration from 'oxidation_config.yaml'
with open('dataset_generation/config_dataset.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Assigning parameters from the configuration file
WINDOW_WIDTH = config['WINDOW_WIDTH']
L_OX_ref = config['L_OX_ref']
ANISO = config['ANISO']
nb_of_vcsel_to_oxydize = config['nb_of_vcsel_to_oxydize']
nb_of_oxyd_iter = config['nb_of_oxyd_iter']
input_dir = config['dir_dataset'] + "/"
speed_depending_on_c = config['speed_depending_on_c']
names_oxydize_aperture = config['names_oxydize_aperture']
list_of_list_L_OX_configs = config['list_of_list_L_OX']

# Convert list_of_list_L_OX_configs to actual speed values
list_of_list_L_OX = []
for list_L_OX in list_of_list_L_OX_configs:
    converted_list_L_OX = [L_OX_ref * speed_depending_on_c[al_concentration] for al_concentration in list_L_OX]
    list_of_list_L_OX.append(converted_list_L_OX)

# Main Script
if __name__ == "__main__":

    for idx_list, list_L_OX in enumerate(list_of_list_L_OX):

        starting_idx = idx_list * nb_of_vcsel_to_oxydize

        for idx, name in enumerate(names_oxydize_aperture):

            for i in range(starting_idx, nb_of_vcsel_to_oxydize + starting_idx):
                # Constants for each VCSEL
                VCSEL_NB = i
                dir_path = f"{input_dir}VCSEL_{VCSEL_NB}/"

                # Directory for saving the output images
                output_directory = os.path.join(dir_path, name)
                os.makedirs(output_directory, exist_ok=True)

                # Loading initial mask and converting to binary
                img = load_png_mask(dir_path, f"init_mask_{name}.png")
                mask_aperture = mask_to_binary(img)

                # Creating linear spaces for aperture dimensions
                x = np.linspace(-WINDOW_WIDTH/2, WINDOW_WIDTH/2, mask_aperture.shape[1])
                y = np.linspace(-WINDOW_WIDTH/2, WINDOW_WIDTH/2, mask_aperture.shape[0])

                # Calculation of the Oxidation Kernel
                kernel_size = 2.1 * list_L_OX[idx]  # Kernel size in micrometers
                aperture_sampling = x[1] - x[0]     # Sampling size for the aperture
                N_kernel = int(np.ceil(kernel_size/aperture_sampling))

                if N_kernel % 2 == 0:
                    N_kernel += 1
                else:
                    pass                
                x_kernel = np.linspace(-kernel_size/2, kernel_size/2, N_kernel)
                y_kernel = np.linspace(-kernel_size/2, kernel_size/2, N_kernel)

                # Conversion to polar coordinates and kernel formation
                R_kernel, theta_kernel = xy_to_rtheta_2D(x_kernel, y_kernel)
                ox_limit = list_L_OX[idx] * (1 - ANISO * np.cos(2 * theta_kernel)**2)

                noyau = (R_kernel <= ox_limit).astype(int)
                noyau = make_the_kernel_odd(noyau)
                noyau = noyau / np.sum(noyau)  # Normalizing the kernel

                # Iteration Process for Oxidation
                t0 = time.time()
                for j in range(nb_of_oxyd_iter):
                    mask_aperture = pad_image(mask_aperture, noyau.size, mode='constant', constant_values=0)
                    mask_aperture = apply_convolution(mask_aperture, noyau)
                    mask_aperture = crop_padded_image(mask_aperture, noyau.size)
                    imageio.imwrite(output_directory + f"/{name}_{j:03d}.png", normalize_and_convert(np.abs(1-mask_aperture)))

                t1 = time.time()
                print(f"Total execution time: {t1 - t0}")
