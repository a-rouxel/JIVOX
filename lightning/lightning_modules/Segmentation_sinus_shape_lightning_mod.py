import pytorch_lightning as pl
import torch
import torch.nn as nn
from OpticalModel import OpticalModel
from AcquisitionModel import AcquisitionModel
from SpectralFilterGenerator import ResNet1D_Peter_with_optim
import segmentation_models_pytorch as smp
import random
from DataModule import VCSELDataModule
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from useful_functions import calculate_mean_spectra
import torch.nn.functional as F
from scipy.ndimage import convolve,binary_dilation
import numpy as np
import os
import time
from pytorch_lightning.profilers import PyTorchProfiler
from datetime import datetime
import io
import pandas as pd
from helper import *




class UnetModel(nn.Module):
    def __init__(self,encoder_name="resnet34",encoder_weights="imagenet",in_channels=1,classes=4,index=0):
        super().__init__()
        self.i= index
        self.model= smp.Unet(encoder_name= encoder_name, in_channels=in_channels,encoder_weights=encoder_weights,classes=classes,activation='sigmoid')
    def forward(self,x):
        x= self.model(x)
        return x

class SegmentationModuleFullNN(pl.LightningModule):
    def __init__(self,num_classes=5,
                        log_images_every_n_steps=400,
                        encoder_weights=None,
                        log_dir="tb_logs",
                        sinus_checkpoint=None,
                        unet_checkpoint=None,
                        learning_rate=1e-4,
                        weight_decay=1e-5,
                        readout_noise_level=0.005,):


        super().__init__()
        self.readout_noise_level = readout_noise_level
        self.optical_model = OpticalModel()
        self.input_spectrum = torch.ones(self.optical_model.num_wavelengths)
        self.acquisition_model = AcquisitionModel(input_spectrum=self.input_spectrum,
                                                  optical_model=self.optical_model,
                                                  auto_exposure=False,
                                                  random_readout_noise=False,
                                                  readout_noise_value=readout_noise_level,
                                                  )
        self.in_channels = 1
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        # self.spectral_filter_generator = SpectralFilterGeneratorLDA()

        num_blocks_per_layer = [3, 2, 4, 2]
        channels_per_block = [32, 64, 128, 128]
        common_dense_size = 256

        self.spectral_filter_generator = ResNet1D_Peter_with_optim(input_channels=5,
                                                                  wavelengths_vec=self.optical_model.wavelengths,
                                                                  num_classes=2,
                                                                  num_blocks_per_layer=num_blocks_per_layer,
                                                                  channels_per_block=channels_per_block,
                                                                  common_dense_size=common_dense_size)



        self.seg_model = UnetModel(classes=num_classes,encoder_weights=encoder_weights,in_channels=self.in_channels)

        if sinus_checkpoint is not None:
            checkpoint = torch.load(sinus_checkpoint, map_location=self.device)
            # a = checkpoint['state_dict'].items()
            # print(a)
            # Adjust the keys
            adjusted_state_dict = {key.replace('spectral_filter_generator.', ''): value
                                     for key, value in checkpoint['state_dict'].items()}

            # remove key from state dict
            # del adjusted_state_dict['wavelengths_vec']
            # Filter out unexpected keys
            model_keys = set(self.spectral_filter_generator.state_dict().keys())
            filtered_state_dict = {k: v for k, v in adjusted_state_dict.items() if k in model_keys}
            self.spectral_filter_generator.load_state_dict(filtered_state_dict, strict=False)


        if unet_checkpoint is not None:
            # Load the weights from the checkpoint into self.seg_model
            checkpoint = torch.load(unet_checkpoint, map_location=self.device)
            # Adjust the keys
            adjusted_state_dict = {key.replace('seg_model.', ''): value
                                   for key, value in checkpoint['state_dict'].items()}
            # Filter out unexpected keys
            model_keys = set(self.seg_model.state_dict().keys())
            filtered_state_dict = {k: v for k, v in adjusted_state_dict.items() if k in model_keys}

            self.seg_model.load_state_dict(filtered_state_dict, strict=False)
        #Freeze the seg_model parameters
        # for param in self.seg_model.parameters():
        #      param.requires_grad = False

        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.f1_score = torchmetrics.F1Score(task="multiclass",num_classes=num_classes)
        self.precision = torchmetrics.Precision(task="multiclass",num_classes=num_classes)
        self.recall = torchmetrics.Recall(task="multiclass",num_classes=num_classes)
        # Example class weights (replace these with your calculated weights)
        class_weights = torch.tensor([0.2, 1, 5, 80.0, 20.0], dtype=torch.float32)
        # If you're using a GPU, remember to transfer weights to the same device as your model
        if torch.cuda.is_available():
            class_weights = class_weights.cuda()
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        # self.rmse = nn.MSELoss()
        self.loss_f1 = nn.MSELoss()
        self.log_images_every_n_steps = log_images_every_n_steps
        
        self.writer = SummaryWriter(log_dir)

    # def setup(self, stage=None):
    #     # Existing setup code for initializing datasets...
    #     # Then, store a sample input
    #     if stage == "fit" or stage is None:
    #         sample_input, _ = next(iter(self.train_dataloader()))
    #         self.sample_input = sample_input[0].unsqueeze(
    #             0)  # Assuming the first item is the input tensor, add batch dimension if necessary
    #
    # def on_fit_start(self):
    #
    #     sample_input = self.sample_input.to(self.device)
    #     writer = SummaryWriter(self.logger.log_dir)
    #     writer.add_graph(self, sample_input)
    #     writer.close()



    def _normalize_data_by_itself(self,data):
        # Calculate the mean and std for each batch individually
        # Keep dimensions for broadcasting
        mean = torch.mean(data, dim=[1,2,3], keepdim=True)
        std = torch.std(data, dim=[1,2,3], keepdim=True)
        
        # Normalize each batch by its mean and std
        normalized_data = (data - mean) / std
        return normalized_data

    def _normalize_data(self, data):
        return (data - self.mean) / self.std
    
    def forward(self, x):
        hyperspectral_cube, spectra, spectra_labels = x

        mean_spectra = calculate_mean_spectra(spectra, spectra_labels)

        # plt.plot(spectra[0,2].detach().cpu().numpy())
        # plt.plot(spectra[0,3].detach().cpu().numpy())
        # plt.plot(spectra[0,4].detach().cpu().numpy())
        # plt.show()

        # plt.plot(spectra[1,2].detach().cpu().numpy())
        # plt.plot(mean_spectra[1,3].detach().cpu().numpy())
        # plt.plot(mean_spectra[1,4].detach().cpu().numpy())
        # plt.show()

        # plt.plot(mean_spectra[2,2].detach().cpu().numpy())
        # plt.plot(mean_spectra[2,3].detach().cpu().numpy())
        # plt.plot(mean_spectra[2,4].detach().cpu().numpy())
        # plt.show()


        # mean_spectra = mean_spectra.view(mean_spectra.shape[0], -1)  # x.size(0) is the batch size

        sinus_params = self.spectral_filter_generator(mean_spectra)
        f_max = 30
        f_min = 12
        phase_min =0
        phase_max = 2*torch.pi
        f_new = (f_max - f_min) / 2 * sinus_params[:, 0] + (f_max + f_min) / 2

        phase_new = (phase_max - phase_min) / 2 * sinus_params[:, 1] + (phase_max + phase_min) / 2
        phase_new2 = phase_new + torch.pi

        x1 = torch.stack((f_new, phase_new), dim=1)
        x2 = torch.stack((f_new, phase_new2), dim=1)


        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        # print("sinus_params ",x_modified)
        # print("x_modified 2", x1.shape)
        spectral_filter1 = self.spectral_filter_generator.generate_sinusoids(x1).float()
        spectral_filter1 = (spectral_filter1 + 1) /2

        spectral_filter2 = self.spectral_filter_generator.generate_sinusoids(x2).float()
        spectral_filter2 = (spectral_filter2 + 1) /2


        self.mean_spectra = mean_spectra
        self.spectral_filter1 = spectral_filter1
        self.spectral_filter2 = spectral_filter2

        self.spectral_filters_list = [spectral_filter1, spectral_filter2]

        image = self.take_images(self.spectral_filters_list, hyperspectral_cube)
        self.image = image

        # freq_sinus_params = sinus_params[:,:,0].unsqueeze(-1)
        return self.seg_model(image)
    
    def take_images(self, spectral_filters_list, hyperspectral_cube):


        spectral_filter1, spectral_filter2 = spectral_filters_list

        image1 = self.acquisition_model(spectral_filter1,hyperspectral_cube)
        image2 = self.acquisition_model(spectral_filter2,hyperspectral_cube)

        # plt.imshow(image1[0,0,:,:].cpu().detach().numpy())
        # plt.colorbar()
        # plt.show()
        # plt.imshow(image2[0,0,:,:].cpu().detach().numpy())
        # plt.colorbar()
        # plt.show()

        result_float = torch.zeros((hyperspectral_cube.shape[0], self.in_channels, hyperspectral_cube.shape[1], hyperspectral_cube.shape[2]))
        result_float[:, 0, :, :] = image1[:,0,:,:] - image2[:,0,:,:]
        result_float = result_float.float().to(self.device)

        # plt.imshow(result_float[0,0,:,:].cpu().detach().numpy())
        # plt.colorbar()
        # plt.show()

        normalized_image = self._normalize_data_by_itself(result_float)

        self.result_float = normalized_image

        return normalized_image

    def plot_spectral_filter(self,spectral_filters_list,mean_spectra):
        batch_size, nb_spectra_filters,spectrum_len = spectral_filters_list[0].shape
        # Create a figure with subplots arranged horizontally
        fig, axs = plt.subplots(1, batch_size, figsize=(batch_size * 5, 4))  # Adjust figure size as needed

        # Check if batch_size is 1, axs might not be iterable
        if batch_size == 1:
            axs = [axs]

        # Plot each spectral filter in its own subplot
        for i in range(batch_size):

            for j in range(len(spectral_filters_list)):
                spec_filter = spectral_filters_list[j].cpu().detach().numpy()[i,0,:]
                spec_filter = spec_filter*0.1 + 0.3
                axs[i].plot(self.optical_model.wavelengths,spec_filter,label="spectral filter nb "+str(j),linestyle='--')
            axs[i].set_title(f"Spectral Filter {i + 1}")
            axs[i].set_xlabel("Wavelength index")
            axs[i].set_ylabel("Filter value")
            axs[i].grid(True)

            for j in range(3,5):
                mean_spec = mean_spectra[i,j,:].cpu().detach().numpy()
            # diff_norm =  mean_spectra[i,4,:].cpu().detach().numpy() - mean_spectra[i,3,:].cpu().detach().numpy().min()
            # diff = diff_norm / diff_norm.max()
            # diff = diff
                axs[i].plot(self.optical_model.wavelengths,mean_spec,label="spectrum nb "+str(j))
            # axs[i].plot(self.optical_model.wavelengths,diff,label="difference 4 and 5")
            # axs[i].plot(self.optical_model.wavelengths,target_sinus[i,0,:].cpu().detach().numpy(),label="target sinus")
        plt.legend()
        # Adjust layout
        plt.tight_layout()

        # Create a buffer to save plot
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)

        # Convert PNG buffer to PIL Image
        image = Image.open(buf)

        # Convert PIL Image to Tensor
        image_tensor = transforms.ToTensor()(image)
        return image_tensor

    def _log_images_grid(self, tag, images, global_step):
        # Assuming images is of shape (batch_size, channels, height, width)
        batch_size, channels, height, width = images.shape
        # Transpose images to shape (channels, batch_size, height, width)
        # Then view it as (channels * batch_size, 1, height, width) for make_grid
        images = images.transpose(0, 1).contiguous().view(channels * batch_size, 1, height, width)
        # Create a grid with images laid out in rows per channel
        # Set nrow to the batch_size to have each row display all batch images for a single channel
        img_grid = torchvision.utils.make_grid(images, nrow=batch_size)
        # Log to TensorBoard
        self.writer.add_image(tag, img_grid, global_step)

    def _log_images(self, tag, images, global_step):
        # Convert model output to image grid and log to TensorBoard
        img_grid = torchvision.utils.make_grid(images)
        self.writer.add_image(tag, img_grid, global_step)

    def _normalize_image_tensor(self, tensor):

        # Normalize the tensor to the range [0, 1]
        min_val = tensor.min()
        max_val = tensor.max()
        normalized_tensor = (tensor - min_val) / (max_val - min_val)
        return normalized_tensor

    def _convert_output_to_images(self, output):
        # Convert the output probabilities to segmentation maps
        # Apply argmax to get the most likely class for each pixel
        segmentation_maps = torch.argmax(output, dim=1)

        # Create a colormap for visualization
        # Assuming you have 4 classes, create a colormap with 4 colors
        # Adjust these colors to match your classes
        colormap = torch.tensor([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],[255, 0, 255]], device=output.device)

        # Map each class index to its corresponding color
        colored_segmentation_maps = colormap[segmentation_maps]

        # Move the color channel to the correct dimension and normalize
        colored_segmentation_maps = colored_segmentation_maps.permute(0, 3, 1, 2).float() / 255.0

        # Create a grid of images for visualization
        img_grid = torchvision.utils.make_grid(colored_segmentation_maps)
        return img_grid
    
    def on_train_epoch_end(self):

        for name, param in self.named_parameters():
            self.writer.add_histogram(name, param, self.current_epoch)

    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx)


        accuracy = self.accuracy(y_hat, y)
        f1_score = self.f1_score(y_hat, y)
        self.log_dict(
            { "train_loss": loss,
              "train_accuracy": accuracy,
                "train_f1_score": f1_score,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )


        # if batch_idx < 100 and self.current_epoch == 0:

        #     if not os.path.exists("./test_images"):
        #         os.makedirs("./test_images")

        #     fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        #     ax[0].imshow(self.result_float[0,0,...].detach().cpu().numpy())
        #     # ax[1].imshow(self.result_float[0, 1, ...].detach().cpu().numpy())
        #     # plt.colorbar()
        #     plt.savefig(f"./test_images/{self.global_step}.svg")

        if batch_idx % self.log_images_every_n_steps == 0:

            predicted_maps = self._convert_output_to_images(y_hat)
            acquired_images = self._normalize_image_tensor(self.result_float)

            y_one_hot = F.one_hot(y, num_classes=5)
            y_one_hot = y_one_hot.permute(0, 3, 1, 2)
            ground_truth_maps = self._convert_output_to_images(y_one_hot)

            self._log_images('train/predicted_maps', predicted_maps, self.global_step)
            self._log_images_grid('train/acquired_images', acquired_images, self.global_step)
            self._log_images('train/ground_truth_maps', ground_truth_maps, self.global_step)
            # Generating and logging the spectral filter plot
            spectral_filter_plot = self.plot_spectral_filter(self.spectral_filters_list,self.mean_spectra)
            self.writer.add_image('Spectral Filter', spectral_filter_plot, self.global_step)

        return {"loss": loss, "scores":y_hat, "y":y}

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(y_hat, y)
        f1_score = self.f1_score(y_hat, y)
        self.log_dict(
            { "val_loss": loss,
              "val_accuracy": accuracy,
              "val_f1_score": f1_score,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        # Check if the current epoch is a multiple of 5
        if (self.current_epoch + 1) % 5 == 0:
            output_images = self._convert_output_to_images(y_hat)
            input_images = self._normalize_image_tensor(self.result_float)
            self._log_images('val/output_images', output_images, self.global_step)
            self._log_images_grid('val/input_images', input_images, self.global_step)

        return {"loss": loss, "scores":y_hat, "y":y}


    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx)

        SNR_indiv = self.acquisition_model.SNR

        # Convert y_hat to class predictions
        y_hat_pred = torch.argmax(y_hat, dim=1)

        # Metrics for Classes 3 & 4 Combined
        y_hat_classes_34 = torch.where((y_hat_pred == 3) | (y_hat_pred == 4), y_hat_pred, torch.tensor(0, device=y_hat_pred.device))
        y_classes_34 = torch.where((y == 3) | (y == 4), y, torch.tensor(0, device=y.device))
        
        # Calculate metrics for classes 3 & 4
        metrics_classes_34 = calculate_metrics(y_hat_classes_34, y_classes_34, label_of_interest=[3, 4])

        # Metrics for All Classes
        metrics_all_classes = calculate_metrics(y_hat_pred, y, label_of_interest=None) # Pass None or all class labels to consider all

        self.log_dict(
            {
                "test_loss": loss,
                **{"class_34_" + k: v for k, v in metrics_classes_34.items()},
                **{"overall_" + k: v for k, v in metrics_all_classes.items()},
                "SNR_indiv": SNR_indiv,
            },
            on_step=False, # Change to True if you want to log each step
            on_epoch=True,
            prog_bar=True,
        )

        if not os.path.exists("./test_images_joint_training"):
            os.makedirs("./test_images_joint_training")

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(self.result_float[0,0,...].detach().cpu().numpy(),cmap="gray")
        ax[1].imshow(y_hat_pred[0, ...].detach().cpu().numpy())
        # plt.colorbar()
        plt.savefig(f"./test_images_joint_training/{batch_idx}_image_{self.readout_noise_level}.svg")
        plt.close()

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        diff_norm = self.mean_spectra[0, 4, :].cpu().detach().numpy() - self.mean_spectra[0, 3, :].cpu().detach().numpy().min()
        diff = diff_norm / diff_norm.max()
        diff = diff
        # for i in range(5):
        #     ax.plot(self.optical_model.wavelengths, self.mean_spectra[0, i, :].cpu().detach().numpy(), label="spectrum nb "+str(i))

        ax.plot(self.optical_model.wavelengths, diff, label="difference 4 and 5")
        ax.plot(self.optical_model.wavelengths,self.spectral_filter1[0,0,:].detach().cpu().numpy())
        ax.plot(self.optical_model.wavelengths,self.spectral_filter2[0,0,:].detach().cpu().numpy())

        ax.set_xlim(0.7, 0.86)
        # ax[1].imshow(self.result_float[0, 1, ...].detach().cpu().numpy())
        # plt.colorbar()
        plt.savefig(f"./test_images_joint_training/{batch_idx}_spectra_{self.readout_noise_level}.svg")
        plt.close()

        return {"loss": loss, "scores": y_hat, "y": y}

    def predict_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx)
        self.log('predict_step', loss,on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_test_end(self):
        metrics = self.trainer.logged_metrics  # This contains all logged metrics
        # convert metrics to float
        metrics = {k: float(v) for k, v in metrics.items()}
        # Convert metrics to a DataFrame or any format you prefer
        metrics_df = pd.DataFrame([metrics])
        # File path
        file_path = './test_results_joint_training.csv'
        # Check if file exists to decide whether to write header
        file_exists = os.path.isfile(file_path)
        # Save to CSV, append if file exists, include header if file does not exist
        metrics_df.to_csv(file_path, mode='a', index=False, header=not file_exists)

    def _common_step(self, batch, batch_idx):
        hyperspectral_cube, spectra,spectra_labels,y, labels_sinus = batch
        x = (hyperspectral_cube, spectra,spectra_labels)
        y_hat = self.forward(x)
        y = y.long()

        loss1 = self.loss_fn(y_hat, y)

        loss =  loss1


        return loss, y_hat, y

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=12, T_mult=2, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler, "interval": "epoch"}}

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
    #     # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=12, T_mult=2, eta_min=1e-6)
    #     return optimizer


        
