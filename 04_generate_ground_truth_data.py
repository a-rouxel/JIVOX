"""
VCSEL Ground Truth Image Generation Script
===========================================

Description:
------------
This script is designed to generate ground truth images for Vertical-Cavity Surface-Emitting Lasers (VCSELs) for specific experiments.
It processes a set number of VCSELs, handling various mask configurations based on the specified experiment type. The script reads binary masks
and stack images, combines them, normalizes the results to produce ground truth images, and saves these images. Additionally, it handles spectral
data associated with each mask, reading from HDF5 files and saving the spectra alongside the ground truth images.

Key Functionalities:
---------------------
1. Experiment-Based Processing:
   - Supports different processing flows based on the specified experiment type.
   - Handles multiple mask configurations and associates spectral data with each mask.

2. Image Processing:
   - Reads, binarizes, and processes mask and stack images for each VCSEL.
   - Combines multiple images to create a ground truth representation for each VCSEL.

3. Ground Truth Generation and Saving:
   - Normalizes the combined images to generate ground truth images.
   - Saves these images in a specified directory.
   - Optionally generates and saves rescaled ground truth images as GIFs.

4. Spectral Data Handling:
   - Reads spectral data from HDF5 files for each mask.
   - Saves the spectral data in the input directory for further analysis.

Author: A.Rouxel
Date: 22/01/2024

Note:
-----
This script is part of a larger suite of tools for VCSEL analysis and assumes the presence of specific directory structures, input file formats, and external modules for handling image and spectral data. Modifications might be required to adapt it to different datasets or analysis requirements.
"""

import imageio
import numpy as np
import os
import glob
import yaml
from dataset_generation.useful_functions import load_png_files, mask_to_binary, binarize_image, generate_gt_images, read_h5_data,save_results_to_h5py_file



with open('dataset_generation/config_dataset.yaml', 'r') as file:
    config = yaml.safe_load(file)


# Set the number of VCSELs to process and other configuration parameters
exps = ["stacks_VCSEL850","stacks_VCSEL940_non_resonnant","stacks_VCSEL940_resonnant"]

# nb_of_vcsel_to_process = 200
nb_of_vcsels_per_exp = config['nb_of_vcsels_per_category']
input_dir = config['dir_dataset']
base_spectra_dir = config['dir_dataset'] + "/output_spectra/"

rescale_factor = config['rescale_factor']
nb_of_frames = config['nb_of_oxyd_iter']

masks_names = [value for key, value in config['object_names'].items()]
is_mask_list= [True, True, False, False,False]



for idx_exp,exp in enumerate(exps):

    # starting_idx = idx_exp
    # Loop through each VCSEL
    for idx in range(nb_of_vcsels_per_exp):

        # Directory path for each VCSEL
        VCSEL_NB = idx  + idx_exp * nb_of_vcsels_per_exp
        dir_path = f"{input_dir}VCSEL_{VCSEL_NB}/"


        print(f"Processing VCSEL {VCSEL_NB}...")
        list_images = []
        

        # Process each mask
        for idx_, masks in enumerate(masks_names):
            images = []
            # Check if it's a mask list or a single mask
            if is_mask_list[idx_]:
                # Load and binarize each image in the mask list
                images = load_png_files(dir_path + f"/{masks_names[idx_]}")
                images = [binarize_image(image) for image in images]
            else:
                # Load and binarize a single image
                images.append(imageio.v2.imread(dir_path + f"init_mask_{masks_names[idx_]}.png"))
                images[0] = binarize_image(mask_to_binary(images[0]))
            list_images.append(images)
        
        # Build the ground truth images
        list_ground_truth = [np.zeros_like(images[0]) for _ in range(nb_of_frames)]
        
        # Combine images to create ground truth for each frame
        for idx__, image in enumerate(list_ground_truth):
            for idx_mask, mask_name in enumerate(masks_names):
        
                try:
                    list_ground_truth[idx__] += list_images[idx_mask][idx__]
                except:
                    list_ground_truth[idx__] += list_images[idx_mask][0]
        
        # Create a directory for saving ground truth images
        output_directory = dir_path + "ground_truth"
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        
        
        list_normalized_ground_truth = []
        # Normalize and save the ground truth images
        for j in range(nb_of_frames):
            normalized_ground_truth = (list_ground_truth[j] * 255 / len(masks_names)).astype(np.uint8)
            list_normalized_ground_truth.append(normalized_ground_truth)
            imageio.imwrite(output_directory + f"/ground_truth_{j:03d}.png", normalized_ground_truth)
        
        
        # Generate and save ground truth images with rescaling as GIF
        generate_gt_images(list_normalized_ground_truth, dir_path, rescale_factor=0.25)

        for nb_frame in range(nb_of_frames):

            # print(nb_frame)
            idx_spectra = idx*nb_of_frames + nb_frame
            # print(idx_spectra)
            spectra_dir = f"{base_spectra_dir}{exp}/stacks_{idx_spectra}/"
            # print(spectra_dir)
            dict_spectra = {}
            # print(spectra_dir)
            # Use glob to find all .h5 files in the specified directory
            h5_files = glob.glob(os.path.join(spectra_dir, "*.h5"))

            list_mask_names = []
            # Loop over the found .h5 files and read their data
            for h5_file_path in h5_files:
                # Extract the mask name from the file name
                mask_name = os.path.basename(h5_file_path).replace(".h5", "")
                dict_spectra[mask_name] = read_h5_data(spectra_dir, mask_name + ".h5", wavelengths=True,
                                                    weighted_spectrum=True)
                list_mask_names.append(mask_name)

            spectra_final_dir = dir_path + "spectra/" + f"{nb_frame}/"
            if not os.path.exists(spectra_final_dir):
                os.makedirs(spectra_final_dir, exist_ok=True)

            #save spectra
            for mask in list_mask_names:
                # print(spectra_dir)
                temp_dict = read_h5_data(spectra_dir, mask + ".h5", wavelengths=True, weighted_spectrum=True)
                # save in input dir
                shape0 = temp_dict["weighted_spectrum"].shape[0]
                if shape0 == 150:
                    print("damn")
                # print(spectra_final_dir)
                save_results_to_h5py_file(spectra_final_dir,f"{mask}.h5", temp_dict)


