import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import os 
import numpy as np
import h5py
from pytorch_lightning import LightningDataModule
import random
from SpectralFilterGenerator import ResNet1D_multiplesinusoids_2,BasicBlock
from OpticalModel import OpticalModel
from AcquisitionModel import AcquisitionModel
from useful_functions import *



class SpectralIlluminationDataset(Dataset):
    
    def __init__(self, data_dir_path):
        """
        Args:
            data_dir_path (list): List of paths to hyperspectral images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dir_path = data_dir_path
        self.data_file_names = [f for f in sorted(os.listdir(self.dir_path)) if f.startswith('data')]
        self.label_file_names = [f.replace('data', 'label') for f in self.data_file_names]

    def __len__(self):
        return len(self.data_file_names)

    def __getitem__(self, idx):
        # Convert
        #  to tensor
        spectra, sinus_labels= self.load_hyperspectral_cube(idx)


        return spectra, sinus_labels


    def load_hyperspectral_cube(self, idx):
        
        file_path = os.path.join(self.dir_path, self.data_file_names[idx])

        with h5py.File(file_path, 'r') as data:
            spectra = data['spectra'][:]
            std = data['std'][:]
            wavelengths = data['wavelengths'][:]
            # Load labels
        label_file_path = os.path.join(self.dir_path, self.label_file_names[idx])
        with h5py.File(label_file_path, 'r') as label_data:
            sinus_labels = label_data['sinus_params'][()]  # Adjust 'label_map' to the actual dataset key
            sinus_labels = sinus_labels.astype(np.float32)


        return spectra, sinus_labels
        


class SpectralIlluminationModule(LightningDataModule):
    def __init__(self,data_dir,batch_size,num_workers=1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = SpectralIlluminationDataset(self.data_dir)

    def setup(self, stage=None):
        dataset_size = len(self.dataset)
        train_size = int(0.7 * dataset_size)
        val_size = int(0.2 * dataset_size)
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


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    data_dir_path = "/data/dataset_images_VCSELs_only_beginning_but_not_full"
    batch_size = 2
    images_dataset = SpectralIlluminationDataset(data_dir_path = data_dir_path)

    optical_model = OpticalModel()
    input_spectrum = torch.ones(optical_model.num_wavelengths)
    acquisition_model = AcquisitionModel(input_spectrum=input_spectrum,optical_model=optical_model)
    
    spectral_filter_generator = ResNet1D_multiplesinusoids_2(BasicBlock,
                                                                    [3,3,3,3],
                                                                    optical_model.wavelengths,
                                                                    num_output_features=2,
                                                                    num_predictions=1,)

    spectra, labels = images_dataset[1]

    # spectra = torch.tensor(spectra).unsqueeze(0)
    # labels_spectra = np.zeros(spectra.shape[1])
    # # fill with labels
    # for i in range(5):
    #     labels_spectra[i * 100: (i + 1) * 100] = i
    # labels_spectra = torch.tensor(labels_spectra).unsqueeze(0)
    #
    # print(spectra.shape, labels_spectra.shape)
    #
    # plt.plot(optical_model.wavelengths, spectra.squeeze(0).numpy()[0,:])
    # plt.plot(optical_model.wavelengths, spectra.squeeze(0).numpy()[100,:])
    # plt.show()
    #
    # mean_spectra = calculate_mean_spectra(spectra, labels_spectra)
    #
    # mean_spectra = mean_spectra.squeeze(0).numpy()
    print(labels)
    spectral_filter = np.sin(2 * np.pi * labels[1] * spectral_filter_generator.wn_vec_unregular + labels[0])

    diff = spectra[4,:]- spectra[3,:]
    diff /= np.max(diff)

    plt.plot(optical_model.wavelengths, spectral_filter)
    plt.plot(optical_model.wavelengths, diff)
    plt.plot(optical_model.wavelengths, spectra[3,:])
    plt.plot(optical_model.wavelengths, spectra[4,:])
    plt.show()
