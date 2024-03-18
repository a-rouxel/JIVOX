import torch
from torch.utils import data
from torchvision import transforms
import numpy as np
import os
# import cv2
from scipy.interpolate import interp1d
from scipy.optimize import minimize


def torch_linear_interp1d(x, y, x_new):
    x = x.float()
    y = y.float()
    x_new = x_new.float()

    batch_size, number_of_spectra, n = x.shape
    _, _, m = x_new.shape

    x_flat = x.view(-1, n)
    x_new_flat = x_new.view(-1, m)

    # Determine if each sequence in x_flat is in ascending order
    ascending_order = x_flat[:, -1] >= x_flat[:, 0]

    # Correctly expand ascending_order for broadcasting
    ascending_order_expanded = ascending_order.view(batch_size, number_of_spectra, 1).expand(-1, -1, n)

    # Use expanded ascending_order for flipping y where necessary
    y_flipped = torch.where(ascending_order_expanded, y, y.flip(dims=[2]))

    # Ensure x_flat is always in ascending order for interpolation
    x_flat = torch.where(ascending_order_expanded.view(-1, n), x_flat, x_flat.flip(dims=[1]))

    idx_right = torch.searchsorted(x_flat, x_new_flat).clamp(min=1, max=n - 1)
    idx_left = idx_right - 1

    x_left = torch.gather(x_flat, 1, idx_left)
    x_right = torch.gather(x_flat, 1, idx_right)
    y_flat = y_flipped.view(-1, n)
    y_left = torch.gather(y_flat, 1, idx_left)
    y_right = torch.gather(y_flat, 1, idx_right)

    slopes = (y_right - y_left) / (x_right - x_left)
    y_new_flat = y_left + slopes * (x_new_flat - x_left)

    y_new = y_new_flat.view(batch_size, number_of_spectra, m)

    return y_new


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

class semantic_dataset(data.Dataset):
    def __init__(self, split = 'test', transform = None):
        self.void_labels = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_labels = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_labels, range(19)))
        self.split = split
        self.img_path = 'testing/image_2/'
        self.mask_path = None
        if self.split == 'train':
            self.img_path = 'training/image_2/'    
            self.mask_path = 'training/semantic/'
        self.transform = transform
        
        self.img_list = self.get_filenames(self.img_path)
        self.mask_list = None
        if self.split == 'train':
            self.mask_list = self.get_filenames(self.mask_path)
        
    def __len__(self):
        return(len(self.img_list))
    
    def __getitem__(self, idx):
        img = cv2.imread(self.img_list[idx])
        img = cv2.resize(img, (1242, 376))
        mask = None
        if self.split == 'train':
            mask = cv2.imread(self.mask_list[idx], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (1242, 376))
            mask = self.encode_segmap(mask)
            assert(mask.shape == (376, 1242))
        
        if self.transform:
            img = self.transform(img)
            assert(img.shape == (3, 376, 1242))
        else :
            assert(img.shape == (376, 1242, 3))
        
        if self.split == 'train':
            return img, mask
        else :
            return img
    
    def encode_segmap(self, mask):
        '''
        Sets void classes to zero so they won't be considered for training
        '''
        for voidc in self.void_labels :
            mask[mask == voidc] = self.ignore_index
        for validc in self.valid_labels :
            mask[mask == validc] = self.class_map[validc]
        return mask
    
    def get_filenames(self, path):
        files_list = list()
        for filename in os.listdir(path):
            files_list.append(os.path.join(path, filename))
        return files_list


def calculate_precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) > 0 else 0


def calculate_recall(tp, fn):
    return tp / (tp + fn) if (tp + fn) > 0 else 0


def calculate_accuracy(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0


def calculate_f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


def calculate_metrics(predictions, targets, label_of_interest=None):
    if label_of_interest is not None:
        # Create masks for predictions and targets for the classes of interest
        pred_mask = torch.zeros_like(predictions, dtype=torch.bool)
        target_mask = torch.zeros_like(targets, dtype=torch.bool)
        for label in label_of_interest:
            pred_mask |= (predictions == label)
            target_mask |= (targets == label)

        predictions_adjusted = torch.where(pred_mask, predictions, torch.tensor(-1))
        targets_adjusted = torch.where(target_mask, targets, torch.tensor(-1))

        tp = ((predictions_adjusted != -1) & (predictions_adjusted == targets_adjusted)).sum().item()
        fp = ((predictions_adjusted != -1) & (targets_adjusted == -1)).sum().item()
        fn = ((predictions_adjusted == -1) & (targets_adjusted != -1)).sum().item()
        tn = ((predictions_adjusted == -1) & (targets_adjusted == -1)).sum().item()
    else:
        # This simplistic method assumes binary classification, needs adjustment for multi-class
        tp = ((predictions == targets)).sum().item()
        unique_labels = targets.unique()
        fp = fn = tn = 0
        for label in unique_labels:
            fp += ((predictions == label) & (targets != label)).sum().item()
            fn += ((predictions != label) & (targets == label)).sum().item()
            tn += ((predictions != label) & (targets != label)).sum().item()

    precision = calculate_precision(tp, fp)
    recall = calculate_recall(tp, fn)
    accuracy = calculate_accuracy(tp, tn, fp, fn)
    f1_score = calculate_f1_score(precision, recall)

    return {"precision": precision, "recall": recall, "accuracy": accuracy, "f1_score": f1_score}

def get_mean_spectra(spectra, nb_of_spectra):
    mean_spectra = np.zeros((nb_of_spectra, spectra.shape[1]))
    for i in range(nb_of_spectra):
        mean_spectra[i, :] = np.mean(spectra[i * 100:(i + 1) * 100, :], axis=0)
        # plt.plot(mean_spectra)
    # plt.show()

    return mean_spectra


import matplotlib.pyplot as plt
def normalize_spectra(mean_spectra):
    diff_last_spectra = mean_spectra[4, :] - mean_spectra[3, :]
    diff_last_spectra -= np.min(diff_last_spectra)
    diff_last_spectra /= np.max(diff_last_spectra)
    diff_last_spectra *= 2
    diff_last_spectra -= 1
    return diff_last_spectra


def sinusoidal_function(x, frequency, phase):
    return np.sin(2 * np.pi * frequency * x + phase)

def create_spectral_filter(mean_spectra,optical_model):


    normalized_spectrum_init = normalize_spectra(mean_spectra)

    # Parameters for moving average
    window_size = 20  # Size of the moving average window
    moving_average = np.convolve(normalized_spectrum_init, np.ones(window_size)/window_size, mode='valid')
    padded_moving_average = np.pad(moving_average, (window_size//2, window_size-1-window_size//2), mode='edge')
    # Subtract the moving average from the normalized spectrum
    normalized_spectrum = normalized_spectrum_init - padded_moving_average

    vector_wn = np.array([1/lambda_ for lambda_ in optical_model.wavelengths])
    vector_wn_regular = np.linspace(vector_wn[0],vector_wn[-1],optical_model.wavelengths.shape[0])

    f = interp1d(vector_wn,normalized_spectrum)

    interpolated_spectrum = f(vector_wn_regular)

    fs = 1/(vector_wn_regular[1]-vector_wn_regular[0])
    # Perform the Fourier Transform
    fft_result = np.fft.fft(normalized_spectrum)
    fft_result_wn = np.fft.fft(interpolated_spectrum)

    # Get the amplitude spectrum
    amplitude_spectrum = np.abs(fft_result)
    amplitude_spectrum2 = np.abs(fft_result_wn)

    # Compute the frequency bins
    n = len(normalized_spectrum)
    frequency_bins = np.fft.fftfreq(n, d=1/fs)


    min_index = 55  # For example, ignore the first few indices to skip the DC component

    # Find the index of the max frequency component, skipping the DC component and starting from min_index
    max_freq_index = np.argmax(amplitude_spectrum2[min_index:]) + min_index  # Adjust index offset
    # print(max_freq_index)
    # Convert the index to actual frequency using frequency bins
    max_freq = frequency_bins[max_freq_index]


    # Objective function to minimize
    def mse(x):
        phi, freq = x
        max_freq_sinusoid2 = np.sin(2 * np.pi * freq * vector_wn_regular + phi)
        f = interp1d(vector_wn_regular, max_freq_sinusoid2)
        reinterpoalted_sinusoid = f(vector_wn)

        rmse = np.sqrt(np.mean((normalized_spectrum - reinterpoalted_sinusoid)**2))
        return rmse

    x = [0, max_freq]

    # Minimize the MSE to find the optimal phi
    result = minimize(mse, x)
    # The optimal phase shift
    phi_optimal = result.x[0] % (2 * np.pi)
    freq_optimal = result.x[1]
    sinus_params = [freq_optimal,phi_optimal]

    return sinus_params