import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pytorch_lightning as pl
from lightning.models.spectral_filter_functions import *
from lightning.models.helper import *
import torch
from torch import nn
import torch.nn.functional as F

def normalize_spectra(mean_spectra):
    diff_last_spectra = mean_spectra[:,4, :] - mean_spectra[:,3, :]
    diff_last_spectra -= torch.min(diff_last_spectra)
    diff_last_spectra /= torch.max(diff_last_spectra)
    diff_last_spectra *= 2
    diff_last_spectra -= 1
    return diff_last_spectra

class SpectralFilterGeneratorLDA(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def generate_filter(self, spectra_array,labels,n_lda_components=2):
        # generate spectral filter
        spectra_array = spectra_array.cpu()
        labels = labels.cpu()
        normalized_lda_coef_in_original_space, eigen = apply_pca_and_lda(spectra_array, labels, 
                                                                         n_pca_components=5, n_lda_components=n_lda_components,
                                                                         delta_lambda=1, window_size=5)
        spectral_filters = generate_spectral_filters(normalized_lda_coef_in_original_space,
                                                    filter_generation_type="LDA")
        

        spectral_filters = [torch.from_numpy(filter).float().to(self.device) for filter in spectral_filters]
    
        return spectral_filters

class SpectralFilterGeneratorSinusoide(pl.LightningModule):

    def __init__(self, wavelengths_vec,nb_of_spectra=4):
        super(SpectralFilterGeneratorSinusoide, self).__init__()

        spectrum_len = wavelengths_vec.shape[0]

        self.input_size = int(spectrum_len * nb_of_spectra)
        self.register_buffer('wavelengths_vec', wavelengths_vec)

        #
        # # Encoder
        # self.encoder = nn.Sequential(
        #     nn.Linear(self.input_size, self.input_size * 2),
        #     nn.ReLU(True),
        #     nn.BatchNorm1d(num_features=self.input_size * 2),
        #     nn.Linear(self.input_size * 2, self.input_size),
        #     nn.ReLU(True),
        #     nn.BatchNorm1d(num_features=self.input_size),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(self.input_size , self.input_size // 2),
        #     nn.ReLU(True),
        #     nn.BatchNorm1d(num_features=self.input_size // 2),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(self.input_size // 2, 256),
        #     nn.ReLU(True),
        #     nn.BatchNorm1d(num_features=256),
        #     nn.Linear(256, 128),
        #     nn.ReLU(True),
        #     nn.BatchNorm1d(num_features=128),
        #     nn.Linear(128, 64),
        #     nn.ReLU(True),
        #     nn.BatchNorm1d(num_features=64),
        #     nn.Linear(64, 32),
        #     nn.ReLU(True),
        #     nn.BatchNorm1d(num_features=32),
        #     nn.Linear(32, 2),
        #     nn.Tanh()
        # )

        # self.conv_layers = nn.Sequential(
        #     # First convolutional block
        #     nn.Conv1d(in_channels=nb_of_spectra, out_channels=16, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm1d(16),
        #     nn.ReLU(True),
        #     nn.Dropout(0.1),  # Reduced dropout rate to retain more information
        #     nn.MaxPool1d(kernel_size=2, stride=2),  # Adding pooling to reduce dimensions
        #
        #     # Second convolutional block (removed the third to simplify)
        #     nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(True),
        #     nn.Dropout(0.1),  # Keeping dropout but at a reduced rate
        #     # Optionally, add a second MaxPool1d here if further reduction is needed
        # )
        #
        # # Calculate reduced spectrum length after convolution and pooling
        # # Adjust this formula based on your actual model architecture
        # reduced_spectrum_len = spectrum_len // 2  # Adjust if you add more pooling layers
        #
        # self.fc_layers = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(32 * reduced_spectrum_len, 256),  # Increase the size here for more complexity
        #     nn.ReLU(True),
        #     nn.Dropout(p=0.3),  # Moderate dropout to prevent overfitting
        #
        #     # Added an intermediate layer for more complexity
        #     nn.Linear(256, 128),
        #     nn.ReLU(True),
        #     nn.Dropout(p=0.3),
        #
        #     nn.Linear(128, 64),
        #     nn.ReLU(True),
        #     nn.Dropout(p=0.3),
        #
        #     nn.Linear(64, 16),
        #     nn.ReLU(True),
        #     nn.Dropout(p=0.3),
        #
        #     nn.Linear(16, 2),
        #     nn.Tanh()
        # )



    def generate_sinusoides(self,periods_offsets):



        """
        Generates a batch of sinusoids based on the given periods and offsets.

        Args:
        - periods_offsets (torch.Tensor): A tensor of shape (batch_size, 2) where each row contains [period, offset].
        - spectrum_len (int): The length of the spectrum to generate for each sinusoid.

        Returns:
        - torch.Tensor: A tensor of shape (batch_size, spectrum_len) containing the generated sinusoids.
        """
        batch_size = periods_offsets.shape[0]
        # Create a tensor representing the x values for the spectrum
        x = self.wavelengths_vec.repeat(batch_size, 1)

        # Calculate the frequency from the period, assuming the period is given in the same units as the x range
        periods = periods_offsets[:, 0].unsqueeze(1)

        # Calculate the phase offset in radians, assuming the offset is directly applicable
        phase_offsets = periods_offsets[:, 1].unsqueeze(1)

        # Generate the sinusoids
        sinusoids = torch.sin((2*torch.pi/((periods+1)/8)) * x + (phase_offsets+1)*torch.pi)

        return sinusoids



    def forward(self, x):
        # x = self.conv_layers(x)
        # x = self.fc_layers(x)
        # x = self.encoder(x)
        x = self.generate_sinusoides(x)
        return x

class SpectralFilterGeneratorNN2(nn.Module):
    def __init__(self, spectrum_len, nb_of_spectra, latent_dim=200):
        super(SpectralFilterGeneratorNN2, self).__init__()

        self.register_buffer('spectrum_len', torch.tensor(spectrum_len))

        # Encoder
        self.conv1 = nn.Conv1d(in_channels=nb_of_spectra, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.drop1 = nn.Dropout(0.25)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.drop2 = nn.Dropout(0.25)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.drop3 = nn.Dropout(0.25)
        # self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(64 * spectrum_len, latent_dim)

        # Decoder
        self.fc2 = nn.Linear(latent_dim, 64 * spectrum_len)
        # self.lstm_inv = nn.LSTM(input_size=128, hidden_size=64, num_layers=2, batch_first=True)
        self.conv4 = nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(32)
        self.drop4 = nn.Dropout(0.25)
        self.conv5 = nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm1d(16)  # Reusing bn3 for simplicity in naming
        self.drop5 = nn.Dropout(0.25)  # Reusing drop3
        self.conv6 = nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm1d(1)
        self.drop6 = nn.Dropout(0.25)

    def encode(self, x):
        x = self.drop1(F.relu(self.bn1(self.conv1(x))))
        x = self.drop2(F.relu(self.bn2(self.conv2(x))))
        x = self.drop3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)  # Flatten for the fully connected layer
        x = torch.tanh(self.fc1(x))
        return x

    def decode(self, z):
        z = F.relu(self.fc2(z))
        z = z.view(-1, 64, self.spectrum_len)  # Reshape for ConvTranspose, adjust according to your spectrum_len
        z = self.drop4(F.relu(self.bn4(self.conv4(z))))
        z = self.drop5(F.relu(self.bn5(self.conv5(z))))
        z = torch.tanh(self.conv6(z))  # Using tanh as the final activation
        return z


    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


class SpectralFilterGeneratorNN(nn.Module):
    def __init__(self, nb_of_spectra=5, spectrum_len = 100,latent_dim=10):
        super(SpectralFilterGeneratorNN, self).__init__()

        nb_of_spectra = torch.tensor(nb_of_spectra)
        spectrum_len = torch.tensor(spectrum_len)
        self.input_size = int(spectrum_len * nb_of_spectra)

        self.register_buffer('spectrum_len', spectrum_len)
        self.register_buffer('nb_of_spectra', nb_of_spectra)


        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, self.input_size),
            nn.ReLU(True),
            nn.BatchNorm1d(num_features=self.input_size),
            nn.Dropout(p=0.5),
            nn.Linear(self.input_size, self.input_size * 2),
            nn.ReLU(True),
            nn.BatchNorm1d(num_features=self.input_size * 2),
            nn.Linear(self.input_size * 2, self.input_size // 2),
            nn.ReLU(True),
            nn.BatchNorm1d(num_features=self.input_size // 2),
            nn.Dropout(p=0.5),
            nn.Linear(self.input_size // 2, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(num_features=256),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.BatchNorm1d(num_features=128),
            nn.Linear(128, spectrum_len),
            nn.Tanh()  # Apply Tanh at the last layer to ensure output is between -1 and 1
        )

        # # Decoder
        # self.decoder = nn.Sequential(
        #     nn.Linear(latent_dim, 32),
        #     nn.ReLU(True),
        #     nn.Linear(32, 64),
        #     nn.ReLU(True),
        #     nn.Linear(64, self.spectrum_len),
        #     nn.ReLU(True),
        #     nn.Linear(self.spectrum_len, self.spectrum_len),
        #     nn.Tanh()  # Apply Tanh at the last layer to ensure output is between -1 and 1
        # )

    def forward(self, x):

        x = self.encoder(x)

        return x
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dropout_rate=0.0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.leaky_relu = nn.LeakyReLU(0.01, inplace=True)  # Using LeakyReLU
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.dropout = nn.Dropout(dropout_rate)  # Adding dropout

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.leaky_relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)  # Applying dropout
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.leaky_relu(out)
        return out

class AttentionPool(nn.Module):
    def __init__(self, input_dim):
        super(AttentionPool, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(self.input_dim, 1)  # Generates a single attention score for each feature

    def forward(self, x):
        # x shape: (batch_size, num_features, sequence_length)
        scores = self.query(x.permute(0, 2, 1))  # Generate attention scores
        scores = F.softmax(scores, dim=1)  # Apply softmax to get probabilities
        # Weighted sum of features with attention scores
        x_att = torch.sum(x * scores.permute(0, 2, 1), dim=2)
        return x_att

# class AttentionPool2(nn.Module):
#     # Placeholder for the AttentionPool class implementation
#     def __init__(self, input_dim):
#         super(AttentionPool2, self).__init__()
#         # Example attention mechanism
#         self.query = nn.Parameter(torch.randn(input_dim, 1))
#
#     def forward(self, x):
#         # Example implementation
#         weights = F.softmax(x.matmul(self.query), dim=1)
#         return torch.sum(weights * x, dim=1)

class ResNet1D(nn.Module):
    def __init__(self, block, num_blocks, wavelengths_vec, num_output_features=2, dropout_rate=0.5):
        super(ResNet1D, self).__init__()

        self.register_buffer('wavelengths_vec', torch.tensor(wavelengths_vec))

        self.in_planes = 64
        self.conv1 = nn.Conv1d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.leaky_relu = nn.LeakyReLU(0.01, inplace=True)  # Using LeakyReLU
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, dropout_rate=dropout_rate)
        self.layer4 = self._make_layer(block, num_output_features, num_blocks[3], stride=2,
                                       dropout_rate=0)  # No dropout in the last layer

        self.attention_pool = AttentionPool(num_output_features)  # Attention-based pooling
        self.tanh = nn.Tanh()
        # Separate heads for frequency and phase offset
        self.freq_head = nn.Linear(num_output_features, 1)  # Adjust as needed
        self.offset_head = nn.Linear(num_output_features, 1)  # Adjust as needed


    def _make_layer(self, block, planes, num_blocks, stride, dropout_rate):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, dropout_rate))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def generate_sinusoides(self,periods_offsets):

        """
        Generates a batch of sinusoids based on the given periods and offsets.

        Args:
        - periods_offsets (torch.Tensor): A tensor of shape (batch_size, 2) where each row contains [period, offset].
        - spectrum_len (int): The length of the spectrum to generate for each sinusoid.

        Returns:
        - torch.Tensor: A tensor of shape (batch_size, spectrum_len) containing the generated sinusoids.
        """
        batch_size = periods_offsets.shape[0]
        # Create a tensor representing the x values for the spectrum
        x = self.wavelengths_vec.repeat(batch_size, 1)

        # Calculate the frequency from the period, assuming the period is given in the same units as the x range
        periods = periods_offsets[:, 0].unsqueeze(1)

        # Calculate the phase offset in radians, assuming the offset is directly applicable
        phase_offsets = periods_offsets[:, 1].unsqueeze(1)

        # Generate the sinusoids
        sinusoids = torch.sin((2*torch.pi/((periods)/8)) * x + (phase_offsets+1)*torch.pi)

        return sinusoids

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attention_pool(x)  # Apply attention-based pooling
        freq_pred = self.freq_head(x)
        offset_pred = self.offset_head(x)
        combined_pred = torch.cat((freq_pred, offset_pred), dim=1)
        return combined_pred

        # return x


class ResNet1D_multiplesinusoids(nn.Module):

    def __init__(self, block, num_blocks, wavelengths_vec, num_output_features=2, num_predictions=1,
                 dropout_rate=0.5):
        super(ResNet1D_multiplesinusoids, self).__init__()

        self.num_predictions = num_predictions  # Number of combined predictions
        self.register_buffer('wn_vec_unregular', torch.tensor([1/lambda_ for lambda_ in wavelengths_vec]))  # Wavenumber vector
        self.register_buffer('wn_vec_regular', torch.linspace(min(self.wn_vec_unregular), max(self.wn_vec_unregular), wavelengths_vec.shape[0]).flip(dims=[0]))  # Regular wavenumber vector
        self.register_buffer('wavelengths_vec', torch.tensor(wavelengths_vec))

        self.in_planes = 64
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.leaky_relu = nn.LeakyReLU(0.01, inplace=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, dropout_rate=dropout_rate)
        self.layer4 = self._make_layer(block, num_output_features, num_blocks[3], stride=2, dropout_rate=0)

        self.attention_pool = AttentionPool(num_output_features)
        self.tanh = nn.Tanh()
        # Adjust heads for N frequency and N phase offset predictions
        self.freq_head = nn.Linear(num_output_features, num_predictions)  # Output N frequencies
        self.offset_head = nn.Linear(num_output_features, num_predictions)  # Output N offsets

    def _make_layer(self, block, planes, num_blocks, stride, dropout_rate):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, dropout_rate))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def generate_sinusoids(self, combined_pred):
        """
        Generates N sinusoids per input sample based on the given frequencies and phase offsets.

        Args:
            combined_pred (torch.Tensor): A tensor of shape (batch_size, num_predictions, 2)
                                          where each slice [i, :, :] contains frequencies and phase offsets
                                          for each prediction.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, num_predictions, spectrum_len) containing
                          the generated sinusoids.
        """

        batch_size, num_sinusoids, _ = combined_pred.shape
        spectrum_len = len(self.wn_vec_regular)

        # Extract frequencies and phase offsets from combined predictions
        periods = combined_pred[:, :, 0]  # shape: (batch_size, num_sinusoids)
        phase_offsets = combined_pred[:, :, 1]  # shape: (batch_size, num_sinusoids)

        # Create a tensor representing the x values for the spectrum, repeated for each sinusoid
        x = self.wn_vec_regular.repeat(batch_size, num_sinusoids,1)  # shape: (batch_size, num_sinusoids, spectrum_len)


        # Calculate the sinusoids
        sinusoids = torch.sin((2 * torch.pi * ((periods+1)/0.1).unsqueeze(-1)) * x + torch.pi*phase_offsets.unsqueeze(-1))


        inverted_sinusoids = torch.flip(sinusoids, dims=[2])
        inverted_wn_regular_vec = self.wn_vec_regular.repeat(batch_size, num_sinusoids,1).flip(dims=[2])
        inverted_wn_unregular_vec = self.wn_vec_unregular.repeat(batch_size, num_sinusoids, 1).flip(dims=[2])


        interpolated_inverted_sinus = torch_linear_interp1d(inverted_wn_regular_vec, inverted_sinusoids, inverted_wn_unregular_vec)
        straight_sinusoids = torch.flip(interpolated_inverted_sinus, dims=[2])


        # plt.plot(self.wavelengths_vec, straight_sinusoids[0,0,:].detach().numpy())
        # plt.plot(self.wavelengths_vec, straight_sinusoids[1,0,:].detach().numpy())
        # plt.show()

        return straight_sinusoids

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attention_pool(x)  # Apply attention-based pooling
        freq_pred = self.tanh(self.freq_head(x))
        offset_pred = self.tanh(self.offset_head(x))
        # Concatenate and then reshape to get the desired output shape
        combined_pred = torch.cat((freq_pred.unsqueeze(-1), offset_pred.unsqueeze(-1)), dim=-1)
        # Ensure the combined predictions are in the shape (batch_size, num_predictions, 2)
        combined_pred = combined_pred.view(-1, self.num_predictions, 2)

        return combined_pred

class ResNet1D_multiplesinusoids_2(nn.Module):

    def __init__(self, block, num_blocks, wavelengths_vec, num_output_features=2, num_predictions=1,
                 dropout_rate=0.5):
        super(ResNet1D_multiplesinusoids_2, self).__init__()

        self.num_predictions = num_predictions  # Number of combined predictions
        self.register_buffer('wn_vec_unregular', torch.tensor([1/lambda_ for lambda_ in wavelengths_vec]))  # Wavenumber vector
        self.register_buffer('wn_vec_regular', torch.linspace(min(self.wn_vec_unregular), max(self.wn_vec_unregular), wavelengths_vec.shape[0]).flip(dims=[0]))  # Regular wavenumber vector
        self.register_buffer('wavelengths_vec', torch.tensor(wavelengths_vec))

        self.in_planes = 64
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.leaky_relu = nn.LeakyReLU(0.01, inplace=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, dropout_rate=dropout_rate)
        self.layer4 = self._make_layer(block, num_output_features, num_blocks[3], stride=2, dropout_rate=0)

        self.attention_pool = AttentionPool(num_output_features)
        self.tanh = nn.Tanh()
        # Adjust heads for N frequency and N phase offset predictions
        self.freq_bn = nn.BatchNorm1d(num_output_features)  # Batch normalization
        self.freq_conv = nn.Conv1d(in_channels=num_output_features, out_channels=num_output_features, kernel_size=3, padding=1, bias=False)  # 1D Convolution
        self.freq_conv_bn = nn.BatchNorm1d(num_output_features)  # Batch normalization after convolution

        self.freq_head = nn.Linear(num_output_features, num_predictions)  # Adjusted to connect after convolution
        self.offset_head = nn.Linear(num_output_features, num_predictions)

    def _make_layer(self, block, planes, num_blocks, stride, dropout_rate):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, dropout_rate))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def generate_sinusoids(self, combined_pred):
        """
        Generates N sinusoids per input sample based on the given frequencies and phase offsets.

        Args:
            combined_pred (torch.Tensor): A tensor of shape (batch_size, num_predictions, 2)
                                          where each slice [i, :, :] contains frequencies and phase offsets
                                          for each prediction.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, num_predictions, spectrum_len) containing
                          the generated sinusoids.
        """

        batch_size, num_sinusoids, _ = combined_pred.shape
        spectrum_len = len(self.wn_vec_regular)

        # Extract frequencies and phase offsets from combined predictions
        periods = combined_pred[:, :, 0]  # shape: (batch_size, num_sinusoids)
        phase_offsets = combined_pred[:, :, 1]  # shape: (batch_size, num_sinusoids)

        # Create a tensor representing the x values for the spectrum, repeated for each sinusoid
        x = self.wn_vec_regular.repeat(batch_size, num_sinusoids,1)  # shape: (batch_size, num_sinusoids, spectrum_len)


        # Calculate the sinusoids
        sinusoids = torch.sin((2 * torch.pi * ((periods+2)/0.1).unsqueeze(-1)) * x + torch.pi*phase_offsets.unsqueeze(-1))


        inverted_sinusoids = torch.flip(sinusoids, dims=[2])
        inverted_wn_regular_vec = self.wn_vec_regular.repeat(batch_size, num_sinusoids,1).flip(dims=[2])
        inverted_wn_unregular_vec = self.wn_vec_unregular.repeat(batch_size, num_sinusoids, 1).flip(dims=[2])


        interpolated_inverted_sinus = torch_linear_interp1d(inverted_wn_regular_vec, inverted_sinusoids, inverted_wn_unregular_vec)
        straight_sinusoids = torch.flip(interpolated_inverted_sinus, dims=[2])


        # plt.plot(self.wavelengths_vec, straight_sinusoids[0,0,:].detach().numpy())
        # plt.plot(self.wavelengths_vec, straight_sinusoids[1,0,:].detach().numpy())
        # plt.show()

        return straight_sinusoids

    def forward(self, x):

        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attention_pool(x)  # Apply attention-based pooling

        freq_pred = self.tanh(self.freq_head(x))
        offset_pred = self.tanh(self.offset_head(x))
        # Concatenate and then reshape to get the desired output shape
        combined_pred = torch.cat((freq_pred.unsqueeze(-1), offset_pred.unsqueeze(-1)), dim=-1)
        # Ensure the combined predictions are in the shape (batch_size, num_predictions, 2)
        combined_pred = combined_pred.view(-1, self.num_predictions, 2)

        return combined_pred

    def extract_max_frequency(self,mean_spectra,nb_of_frequencies=1):

        batch_size, num_spectra, len_spectra = mean_spectra.shape

        normalized_spectrum = normalize_spectra(mean_spectra)
        vector_wn = self.wn_vec_unregular.repeat(batch_size,1, 1).flip(dims=[2])
        wn_vec_regular = self.wn_vec_regular.repeat(batch_size,1, 1)

        # print(vector_wn.shape)
        # print(normalized_spectrum.shape)
        # print(wn_vec_regular.shape)

        interpolated_spectrum = torch_linear_interp1d(vector_wn, normalized_spectrum, wn_vec_regular)

        fft_result = torch.fft.fft(normalized_spectrum)
        fft_result_wn = torch.fft.fft(interpolated_spectrum.float())  # Assuming interpolated_spectrum is a numpy array

        amplitude_spectrum = torch.abs(fft_result)
        amplitude_spectrum2 = torch.abs(fft_result_wn)

        min_index = 5
        max_index = len_spectra // 2

        # Find the indices of the n highest frequencies
        values, max_freq_indices = torch.topk(amplitude_spectrum2[:,:,min_index:max_index], nb_of_frequencies, dim=2, largest=True,
                                              sorted=True)
        # print("max_freq_indices",max_freq_indices.shape)
        # print(values.device, max_freq_indices.device)  # Should output 'cuda:0' if your tensor was on GPU
        max_freq_indices += min_index  # Adjust for the offset

        # Prepare frequency bins
        n = normalized_spectrum.size(1)
        fs = 1 / (wn_vec_regular[0, 0, 1] - wn_vec_regular[0, 0, 0]).item()
        frequency_bins = torch.fft.fftfreq(n, d=1 / fs)
        frequency_bins.to(normalized_spectrum.device)

        # print(max_freq_indices.cpu().shape)
        # print(frequency_bins.shape)
        # print(amplitude_spectrum2.device)  # Check device of the tensor

        # No manual repeat needed for frequency_bins if it's broadcastable
        # Expand frequency_bins to match the shape of max_freq_indices for direct indexing
        expanded_frequency_bins = frequency_bins.repeat(batch_size, 1, 1)
        expanded_frequency_bins = expanded_frequency_bins.to(max_freq_indices.device)

        # print(expanded_frequency_bins.shape, max_freq_indices.shape)
        max_freq = torch.gather(expanded_frequency_bins, 2, max_freq_indices)
        max_freq = torch.abs(max_freq)
        max_freq = max_freq/10 -2

        return max_freq


class ResBlockPeter(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, convblock=False):
        super(ResBlockPeter, self).__init__()
        self.convblock = convblock

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        if self.convblock:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.convblock:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet1D_Peter(nn.Module):
    def __init__(self, input_channels, wavelengths_vec,num_classes=6):
        super(ResNet1D_Peter, self).__init__()

        self.register_buffer('wn_vec_unregular', torch.tensor([1/lambda_ for lambda_ in wavelengths_vec]))  # Wavenumber vector
        self.register_buffer('wn_vec_regular', torch.linspace(self.wn_vec_unregular[0], self.wn_vec_unregular[-1], wavelengths_vec.shape[0]))  # Regular wavenumber vector
        self.register_buffer('wavelengths_vec', torch.tensor(wavelengths_vec))


        self.layer1 = nn.Sequential(
            ResBlockPeter(input_channels, 32, convblock=True),
            ResBlockPeter(32, 32),
            ResBlockPeter(32, 32),
            nn.MaxPool1d(2)
        )
        self.layer2 = nn.Sequential(
            ResBlockPeter(32, 64, convblock=True),
            ResBlockPeter(64, 64),
            ResBlockPeter(64, 64),
            nn.MaxPool1d(2)
        )

        # Adding more layers
        self.layer3 = nn.Sequential(
            ResBlockPeter(64, 128, convblock=True),
            ResBlockPeter(128, 128),
            nn.BatchNorm1d(128),  # Additional regularization
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.layer4 = nn.Sequential(
            ResBlockPeter(128, 256, convblock=True),
            ResBlockPeter(256, 256),
            nn.BatchNorm1d(256),  # Additional regularization
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Adaptive pooling to a fixed size
        )

        # Continue with layer3 and layer4 following the pattern

        # Final layers
        self.flatten = nn.Flatten()
        # Adjust this linear layer size according to the flattened output size from the last pooling/conv layer
        self.common_dense = nn.Linear(256, 128)  # Example, adjust '1600' based on actual size

        # Frequency head
        self.freq_dense = nn.Linear(128, 1)  # Outputs a single value for frequency

        # Phase head
        self.phase_dense = nn.Linear(128, 1)  # Outputs a single value for phase

        # Optional: Activation functions, if needed
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # Continue processing through layers
        x = self.flatten(x)
        x = self.common_dense(x)  # Common processing before splitting into heads

        # Process through the frequency head
        freq = self.freq_dense(x)
        freq = self.tanh(freq)  # Assuming you want to use tanh activation

        # Process through the phase head
        phase = self.phase_dense(x)
        phase = self.tanh(phase)  # Assuming you want to use tanh activation

        # Combine the outputs from both heads
        # Note: This combines them into a single tensor of shape (batch_size, 2)
        output = torch.cat((freq, phase), dim=1)

        # print("output grad",output.grad)


        return output

    def generate_sinusoids(self, predicitions):
        """
        Generates N sinusoids per input sample based on the given frequencies and phase offsets.

        Args:
            combined_pred (torch.Tensor): A tensor of shape (batch_size, num_predictions, 2)
                                          where each slice [i, :, :] contains frequencies and phase offsets
                                          for each prediction.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, num_predictions, spectrum_len) containing
                          the generated sinusoids.
        """

        batch_size, num_sinusoids, _ = predicitions.shape

        # Extract frequencies and phase offsets from combined predictions
        frequencies = predicitions[:, :, 0]  # shape: (batch_size, num_sinusoids)
        phase_offsets = predicitions[:, :, 1]  # shape: (batch_size, num_sinusoids)

        # Create a tensor representing the x values for the spectrum, repeated for each sinusoid
        wn_vec_regular = self.wn_vec_regular.repeat(batch_size, num_sinusoids,1)  # shape: (batch_size, num_sinusoids, spectrum_len)
        wn_vec_unregular = self.wn_vec_unregular.repeat(batch_size, num_sinusoids, 1)  # shape: (batch_size, num_sinusoids, spectrum_len)
        # Calculate the sinusoids
        sinusoids = torch.sin(2 * torch.pi * frequencies * wn_vec_regular + phase_offsets)
        # print("frequencies",frequencies[0,:])
        # print("phase offsets",phase_offsets[0,:])

        # inverted_sinusoids = torch.flip(sinusoids, dims=[2])
        # inverted_wn_regular_vec = self.wn_vec_regular.repeat(batch_size, num_sinusoids,1).flip(dims=[2])
        # inverted_wn_unregular_vec = self.wn_vec_unregular.repeat(batch_size, num_sinusoids, 1).flip(dims=[2])
        # wn_unregular_vec = self.wn_vec_unregular.repeat(batch_size, num_sinusoids, 1)

        # print("inverted regular",inverted_wn_regular_vec[0,0,:])
        # print("inverted unregular",inverted_wn_unregular_vec[0,0,:])

        interpolated_inverted_sinus = torch_linear_interp1d(wn_vec_regular, sinusoids, wn_vec_unregular)
        # straight_sinusoids = torch.flip(interpolated_inverted_sinus, dims=[2])


        # plt.plot(self.wavelengths_vec, straight_sinusoids[0,0,:].detach().numpy())
        # plt.plot(self.wavelengths_vec, straight_sinusoids[1,0,:].detach().numpy())
        # plt.show()

        # print(straight_sinusoids.shape)

        return interpolated_inverted_sinus




class ResNet1D_Peter_with_optim(nn.Module):
    def __init__(self, input_channels, wavelengths_vec, num_classes=6, num_blocks_per_layer=None, channels_per_block=None, common_dense_size=128):
        super(ResNet1D_Peter_with_optim, self).__init__()

        if num_blocks_per_layer is None:
            num_blocks_per_layer = [3, 3, 2, 2]  # Default: 3 blocks for layer1 and layer2, 2 blocks for layer3 and layer4
        if channels_per_block is None:
            channels_per_block = [32, 64, 128, 256]  # Default channels configuration

        self.register_buffer('wn_vec_unregular', torch.tensor([1/lambda_ for lambda_ in wavelengths_vec]))
        self.register_buffer('wn_vec_regular', torch.linspace(self.wn_vec_unregular[0], self.wn_vec_unregular[-1], wavelengths_vec.shape[0]))
        self.register_buffer('wavelengths_vec', torch.tensor(wavelengths_vec))

        layers = []
        input_channel = input_channels
        for layer_idx, num_blocks in enumerate(num_blocks_per_layer):
            layer_channels = channels_per_block[layer_idx]
            layers.append(self._make_layer(input_channel, layer_channels, num_blocks))
            input_channel = layer_channels

        self.layers = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        # Assuming you know the input size to your network, for example:
        input_size = (input_channels, 100)  # You need to specify the correct size

        # Dynamically calculate the correct input size for the dense layer
        correct_dense_input_size = self._calculate_flatten_size(input_size)
        self.common_dense = nn.Linear(correct_dense_input_size, common_dense_size)

        self.freq_dense = nn.Linear(common_dense_size, 1)
        self.phase_dense = nn.Linear(common_dense_size, 1)
        self.tanh = nn.Tanh()

    def _calculate_flatten_size(self, input_size):
        with torch.no_grad():
            x = torch.zeros(1, *input_size)  # Mock input tensor based on the expected input size
            x = self.layers(x)
            return x.view(1, -1).size(1)

    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = [ResBlockPeter(in_channels, out_channels, convblock=True)]
        for _ in range(1, num_blocks):
            layers.append(ResBlockPeter(out_channels, out_channels))
        layers.append(nn.MaxPool1d(2))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        x = self.flatten(x)
        x = self.common_dense(x)

        freq = self.tanh(self.freq_dense(x))
        phase = self.tanh(self.phase_dense(x))
        output = torch.cat((freq, phase), dim=1)

        return output

    def generate_sinusoids(self, predicitions):

        batch_size, num_sinusoids, _ = predicitions.shape

        # Extract frequencies and phase offsets from combined predictions
        frequencies = predicitions[:, :, 0]  # shape: (batch_size, num_sinusoids)
        phase_offsets = predicitions[:, :, 1]  # shape: (batch_size, num_sinusoids)

        wn_vec_regular = self.wn_vec_regular.repeat(batch_size, num_sinusoids,1)  # shape: (batch_size, num_sinusoids, spectrum_len)
        wn_vec_unregular = self.wn_vec_unregular.repeat(batch_size, num_sinusoids, 1)  # shape: (batch_size, num_sinusoids, spectrum_len)

        sinusoids = torch.sin(2 * torch.pi * frequencies * wn_vec_regular + phase_offsets)


        interpolated_inverted_sinus = torch_linear_interp1d(wn_vec_regular, sinusoids, wn_vec_unregular)

        return interpolated_inverted_sinus

if __name__ == "__main__":

    model = resnet18()

    def print_model_parameters(model):
        print("Model's parameters and their shapes:")
        total_params = 0
        for name, param in model.named_parameters():
            print(f"{name}, shape: {param.size()}")
            total_params += param.numel()  # param.numel() returns the total number of elements in the parameter
        print(f"\nTotal number of parameters: {total_params}")

    print_model_parameters(model)