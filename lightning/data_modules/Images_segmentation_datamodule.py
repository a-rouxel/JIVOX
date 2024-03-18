import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import os 
import numpy as np
import h5py
from pytorch_lightning import LightningDataModule
import random

def moving_average(x, w):
    pad_size = w // 2
    # Pad with the edge values
    x_padded = np.pad(x, (pad_size, pad_size), 'edge')
    # Perform convolution on the padded array
    convolved = np.convolve(x_padded, np.ones(w), 'valid') / w
    return convolved
import time 
### DATASET STRUCTURE
### --- IMAGES ONLY ---
# -- labels folder
# * example file : mask_0.png with 0,1,2,...
# -- data folder
# * example file :

def transform_map(input_map, nb_replicas):
    # Deduce the number of classes from the map
    n_class = input_map.max() + 1

    # Randomly select a replica for each class index
    replica_offsets = np.random.randint(0, nb_replicas, size=input_map.shape)

    # Multiply each class index by nb_replicas and add the random offset
    output_map = input_map * nb_replicas + replica_offsets

    return output_map

class VCSELDataset(Dataset):
    
    def __init__(self, data_dir_path, augment=True):
        """
        Args:
            data_dir_path (list): List of paths to hyperspectral images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dir_path = data_dir_path
        self.augment = augment
        self.data_file_names = [f for f in sorted(os.listdir(self.dir_path)) if f.startswith('data')]
        self.label_file_names = [f.replace('data', 'label') for f in self.data_file_names]

    def __len__(self):
        return len(self.data_file_names)

    def __getitem__(self, idx):
        # Convert to tensor
        hyperspectral_cube, spectra, noisy_spectra_labels, labels, labels_sinus = self.load_hyperspectral_cube(idx)

        # Apply any additional transformations (e.g., normalization)
        if self.augment:
            hyperspectral_cube, labels = self.augment_data(hyperspectral_cube,labels)

        hyperspectral_cube = torch.from_numpy(hyperspectral_cube)


        return hyperspectral_cube, spectra, noisy_spectra_labels, labels, labels_sinus

    def augment_data(self, hyperspectral_cube,labels):

        # Randomly decide whether to apply each flip
        if random.random() > 0.5:
            hyperspectral_cube = np.flip(hyperspectral_cube, axis=1)  # Flip along width
            labels = np.flip(labels, axis=1)  # Flip labels along width

        if random.random() > 0.5:
            hyperspectral_cube = np.flip(hyperspectral_cube, axis=0)  # Flip along height
            labels = np.flip(labels, axis=0)  # Flip labels along height

        # Check if the array has negative stride and copy if necessary
        if np.any(np.array(hyperspectral_cube.strides) < 0):
            hyperspectral_cube = hyperspectral_cube.copy()

        # Similarly, for labels
        if np.any(np.array(labels.strides) < 0):
            labels = labels.copy()

        return hyperspectral_cube,labels

    def load_hyperspectral_cube(self, idx):
        
        file_path = os.path.join(self.dir_path, self.data_file_names[idx])
        with h5py.File(file_path, 'r') as data:
            spectra = data['spectra'][:]
            std = data['std'][:]
            wavelengths = data['wavelengths'][:]
            map = data['image'][:]
            map = map.astype(np.uint16)
            transformed_map = transform_map(map, nb_replicas=100)
            hyperspectral_cube, noisy_spectra, labels_noisy_spectra  = self.generate_hyperspectral_cube(wavelengths,spectra, std,transformed_map)
            # Load labels
        label_file_path = os.path.join(self.dir_path, self.label_file_names[idx])
        with h5py.File(label_file_path, 'r') as label_data:
            labels = label_data['image'][:]  # Adjust 'label_map' to the actual dataset key
            labels_sinus = label_data['sinus_params'][:]
        return hyperspectral_cube, noisy_spectra, labels_noisy_spectra, labels,labels_sinus
        
    def generate_noisy_spectra(self,wavelengths,mean_spectra,std,nb_replicas=100):

        nb_of_spectra = mean_spectra.shape[0]
        # Create an empty array for the noisy spectra
        noisy_spectra = np.empty((nb_of_spectra * nb_replicas, *mean_spectra.shape[1:]))
        labels = np.zeros(noisy_spectra.shape[0])
        #fill with labels
        for i in range(mean_spectra.shape[0]):
            labels[i * nb_replicas : (i+1) * nb_replicas] = i


        # For each spectrum in mean_spectra, create 100 noisy replicas
        for i in range(nb_of_spectra):
            for j in range(nb_replicas):
                noisy_spectra[i * nb_replicas + j] =  moving_average(mean_spectra[i] + np.random.normal(0, std[i,:]),11)
        #     plt.plot(wavelengths,mean_spectra[i])
        #     plt.fill_between(wavelengths, mean_spectra[i] - std[i,:], mean_spectra[i] + std[i,:], alpha=0.8)
        # plt.show()

        
        # plt.plot(noisy_spectra[0,:],label='mean spectra renewal')
        # plt.legend()
        # plt.show()

        return noisy_spectra,labels

    def generate_hyperspectral_cube(self, wavelengths,mean_spectra, std, map):
        
        noisy_spectra,labels_noisy_spectra = self.generate_noisy_spectra(wavelengths,mean_spectra,std,nb_replicas=100)

        # Ensure map values are within valid range
        map = np.clip(map, 0, noisy_spectra.shape[0] - 1)

        # Index noisy spectra with the map to generate the hyperspectral cube
        hyperspectral_cube = noisy_spectra[map]

 

        return hyperspectral_cube, noisy_spectra, labels_noisy_spectra


class VCSELDataModule(LightningDataModule):
    def __init__(self,data_dir,batch_size,num_workers=1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = VCSELDataset(self.data_dir,augment=True)

    def setup(self, stage=None):
        dataset_size = len(self.dataset)
        train_size = int(0.05* dataset_size)
        val_size = int(0.05 * dataset_size)
        test_size = dataset_size - train_size - val_size

        self.train_ds, self.val_ds, self.test_ds = random_split(self.dataset, [train_size, val_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            shuffle=False)

# #Testing the VCSELDataset
# data_dir_path = '../dataset_images/'  # Replace with your path
# vcseldataset = VCSELDataset(data_dir_path)

# hyperspectral_cube_0, spectra, labels_noisy, labels = vcseldataset[20]

# datamodule = VCSELDataModule(data_dir_path, batch_size=1, num_workers=4)



