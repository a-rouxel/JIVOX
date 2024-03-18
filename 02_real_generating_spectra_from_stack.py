"""
VCSEL Spectra Generation Script
================================

This script is designed for generating and analyzing the reflectance spectra of Vertical-Cavity Surface-Emitting Lasers (VCSELs).
It utilizes the S-algorithm and a stack-based approach to model the multilayer structures of VCSELs and computes their Reflectivity.
The script integrates optical modeling of the Acquisition system to evaluate how different stack configurations affect the VCSEL's performance.

Note : The script is designed to work with the 'OpticalModel' class and related functions. It takes into account the different angles of incidence related to
the numerical aperture of the microscope objective

Functionality:
--------------
1. Loading and Processing VCSEL Stacks:
   - Reads stack configurations from a JSON file.
   - Processes each stack by adding layers and converting them into refractive index representations.

2. Optical System Modeling:
   - Utilizes the 'OpticalModel' class to set up the optical environment.
   - Computes reflectance and transmittance for each stack at various angles.

3. Spectra Weighting and Aggregation:
   - Applies coefficients from the optical model to weight the calculated spectra.
   - Aggregates these weighted spectra to derive a cumulative spectrum for each stack.

4. Data Visualization and Output:
   - Optionally, Plots the aggregated spectra for visual analysis.
   - Saves the computed spectra and stack details to HDF5 files for further analysis.

Usage:
The script is executed with a predefined stack name and output directory. It offers the option to visualize all generated spectra in a single plot for comparative analysis.

Note:
The script assumes the presence of an external module for the OpticalModel and related functions. It requires the 'matplotlib' and 'numpy' libraries for computation and visualization.

Author: A. Rouxel
Date: 22/01/2024

"""

# Import necessary modules
from dataset_generation.multilayer_stacks.functions_spectra_from_stacks import main
import yaml

# Entry point of the script
if __name__ == '__main__':
    # Define the stack name and output directory
    #stack_names = ["stacks_double_ouverture_v1_eq_v2", "stacks_double_ouverture_v1_sup_v2","stacks_double_ouverture_v1_sup_sup_v2"]
    stack_names = ["stacks_VCSEL940_resonnant", "stacks_VCSEL850","stacks_VCSEL940_non_resonnant"]
    with open('dataset_generation/config_dataset.yaml', 'r') as file:
        config = yaml.safe_load(file)
    dir_configs = config['dir_dataset']
    for stack_name in stack_names:
        input_dir = dir_configs + '/output_spectra/configs/' + stack_name
        output_dir = dir_configs + '/output_spectra/' + stack_name

        # Run the main function with specified parameters
        main(stack_name, input_dir,output_dir, plot_all=False)
