import os
import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import h5py


def load_png_files(directory):
    images = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            image = imageio.v2.imread(image_path)
            images.append(image)
    return images

def mask_to_binary(mask):
    mask_ = mask[:,:,3] > 0
    return mask_

def binarize_image(image):
    image = np.where(image > 0, 1, 0)
    return image


def calculate_mean_spectra(spectra_array, labels):
    batch_size, _, spectrum_len = spectra_array.shape
    max_labels_per_batch = 5  # Adjust based on your needs
    mean_spectra_list = []

    # Process each batch
    for i in range(batch_size):
        batch_spectra = spectra_array[i]  # Shape: (500, spectrum_len)
        batch_labels = labels[i]  # Shape: (500,)
        unique_labels = torch.unique(batch_labels)
        batch_mean_spectra = []

        # Compute mean spectra for up to 5 unique labels
        for label in unique_labels[:max_labels_per_batch]:  # Limiting to first N labels
            mask = batch_labels == label
            mean_spectrum = batch_spectra[mask].mean(dim=0)
            batch_mean_spectra.append(mean_spectrum)

        # Ensure each batch has N mean spectra
        while len(batch_mean_spectra) < max_labels_per_batch:
            # Append zeros or handle as needed
            batch_mean_spectra.append(torch.zeros(spectrum_len, dtype=batch_spectra.dtype, device=batch_spectra.device))

        # Stack mean spectra for the current batch
        batch_mean_spectra_tensor = torch.stack(batch_mean_spectra)
        mean_spectra_list.append(batch_mean_spectra_tensor)

    # Stack all batches together
    mean_spectra = torch.stack(mean_spectra_list).float()  # Shape: (batch_size, max_labels_per_batch, spectrum_len)

    return mean_spectra

def generate_gt_images(list_normalized_ground_truth,dir_path,rescale_factor=0.25):

    modified_images = []

    for i, arr in enumerate(list_normalized_ground_truth):
        print(type(arr))
        # Convert the array to a NumPy array if it's not already one
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)

        # resize image
        print((int(arr.shape[0]*rescale_factor), int(arr.shape[1]*rescale_factor)))
        arr = np.array(Image.fromarray(arr).resize((int(arr.shape[0]*rescale_factor), int(arr.shape[1]*rescale_factor))))

        # Convert array to PIL Image
        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)

        # You can choose a font and size here
        font = ImageFont.load_default()

        # Position of the text
        text_position = (50, 50)  # Change as per your requirement

        # Drawing text on image
        draw.text(text_position, f"Iteration: {i}", fill=(120), font=font)

        # Append the modified image to the list
        modified_images.append(img)

    # Save the images as a GIF
    imageio.mimsave(dir_path + 'ground_truth.gif', modified_images, duration=0.8, format='GIF', loop=0)

def save_results_to_h5py_file(dir_name,filename, data_dict):

    os.makedirs(dir_name, exist_ok=True)

    with h5py.File(dir_name + filename, 'w') as file:
        for key in data_dict.keys():
            file.create_dataset(key, data=data_dict[key])

def read_h5_data(directory, filename, **kwargs):
    """
    Open, read, and visualize specific datasets from an HDF5 file based on the given keyword arguments.

    Parameters:
    - directory (str): Name of the directory where the file is saved.
    - filename (str): Name of the HDF5 file to read from.
    - kwargs: Keyword arguments where each key is the name of a dataset to read. The values are ignored.

    Returns:
    - A dictionary containing the requested datasets.
    """
    filepath = os.path.join(directory, filename)
    data = {}


    with h5py.File(filepath, 'r') as f:

        for dataset in kwargs:
            if dataset in f:
                try:
                    data[dataset] = f[dataset][:]
                except:
                    data[dataset]  = f[dataset][()]

            else:
                print(f"Dataset {dataset} not found in file.")

    return data

def convert_image(image):    # Find unique values in the image and their indices
    unique_values, inverse_indices = np.unique(image, return_inverse=True)

    # Use inverse indices to map each original value to its corresponding integer
    converted_image = inverse_indices.reshape(image.shape).astype(np.uint8)

    return converted_image



def define_rdn_spectrum_std(wavelengths,nb_of_std_vec):

    list_std = []

    for i in range(nb_of_std_vec):
        # Define a list of trigonometric functions
        trig_functions = [np.sin, np.cos, lambda x: np.sin(x) * np.cos(x)]

        # Randomly select one of the trigonometric functions
        chosen_function = random.choice(trig_functions)

        baseline = np.abs(np.random.randn())
        std = baseline*np.ones_like(wavelengths)
        list_std.append(std)

    return list_std