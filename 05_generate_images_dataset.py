import os
import json
import numpy as np
import matplotlib.pyplot as plt
import h5py
import yaml
from useful_functions import read_h5_data, convert_image


with open('dataset_generation/config_dataset.yaml', 'r') as file:
    config_dataset = yaml.safe_load(file)

with open('dataset_generation/config_images_dataset.yaml', 'r') as file:
    config_img_dataset = yaml.safe_load(file)



# Define the path to the VCSEL folder
vcsel_folder = config_dataset['dir_dataset']
image_dataset_dir = config_img_dataset['dir_dataset']
spectra_names = [value for key, value in config_dataset['object_names'].items()]


os.makedirs(image_dataset_dir, exist_ok=True)
counter = 0

# Loop through all directories in the VCSEL folder
for dir_name in sorted(os.listdir(vcsel_folder)):
    dir_path = os.path.join(vcsel_folder, dir_name)
    print(dir_path)
    # Check if it's a directory
    if os.path.isdir(dir_path):

        # 2. Load the corresponding config.json file
        config_path = os.path.join(dir_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as config_file:
                config = json.load(config_file)
        else:
            config = None

        # 3. Load all the png images in the ground_truth subdirectory
        ground_truth_path = os.path.join(dir_path, "ground_truth")
        if os.path.exists(ground_truth_path):
           images = [img for img in sorted(os.listdir(ground_truth_path)) if img.endswith(".png")]
           images =  [np.array(plt.imread(os.path.join(ground_truth_path, img))) for img in images]

           images = [convert_image(img) for img in images]
           images = [img[::4,::4] for img in images]

        else:
            images = []
        
        for idx, im in enumerate(images):

            if idx >3 and idx < 20:

                print(dir_path, f"spectra/{idx}/")
                spectra_dir_VCSEL = os.path.join(dir_path, f"spectra/{idx}/")
                try:
                    spectra_dict = [read_h5_data(spectra_dir_VCSEL, name + ".h5", wavelengths=True, weighted_spectrum=True) for name in
                                    spectra_names]
                except:
                    print(f"Error reading {spectra_dir_VCSEL}")
                    break

                spectra = [np.real(spectrum["weighted_spectrum"]) for spectrum in spectra_dict]
                wavelengths = spectra_dict[0]["wavelengths"]

                std = [np.ones_like(wavelengths) * np.abs(np.random.randn()) * 0.002 for spectrum in spectra_dict]


                # Create an HDF5 file for this directory
                with h5py.File(os.path.join(image_dataset_dir, "data_{:06d}.h5".format(counter)), 'w') as h5f:
                    # Save spectra
                    h5f.create_dataset('spectra', data=spectra)
                    h5f.create_dataset('std', data=std)
                    h5f.create_dataset('wavelengths', data=wavelengths)
                    h5f.create_dataset('config', data=json.dumps(config))
                    h5f.create_dataset(f'image', data=im)

                with h5py.File(os.path.join(image_dataset_dir, "label_{:06d}.h5".format(counter)), 'w') as h5f:
                    #save ground truth
                    h5f.create_dataset('image', data=im)
                counter += 1




