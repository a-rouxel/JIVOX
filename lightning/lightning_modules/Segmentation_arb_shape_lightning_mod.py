import pytorch_lightning as pl
import torch.nn as nn
from physics_models.OpticalModel import OpticalModel
from physics_models.AcquisitionModel import AcquisitionModel
from lightning.models.SpectralFilterGenerator_model import SpectralFilterGeneratorLDA
import segmentation_models_pytorch as smp
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
import torchvision
from lightning.models.helper import *
import pandas as pd
import io
from PIL import Image
from torch.nn import functional as F
from lightning.lightning_modules.useful_functions import *


class UnetModel(nn.Module):
    def __init__(self,encoder_name="resnet34",encoder_weights="imagenet",in_channels=1,classes=4,index=0):
        super().__init__()
        self.i= index
        self.model= smp.Unet(encoder_name= encoder_name, in_channels=in_channels,encoder_weights=encoder_weights,classes=classes,activation='sigmoid')
    def forward(self,x):
        x= self.model(x)
        return x

class SegmentationModule(pl.LightningModule):
    def __init__(self,num_classes=5,
                 log_images_every_n_steps=1,
                 encoder_weights="imagenet",
                 log_dir="tb_logs",
                 unet_checkpoint="",
                 readout_noise_level = 0.0005):
        super().__init__()
        self.readout_noise_level = readout_noise_level
        self.optical_model = OpticalModel()
        self.input_spectrum = torch.ones(self.optical_model.num_wavelengths)
        self.acquisition_model = AcquisitionModel(input_spectrum=self.input_spectrum,
                                                  optical_model=self.optical_model,
                                                  auto_exposure=False,
                                                  random_readout_noise=False,
                                                  readout_noise_value = readout_noise_level,
                                                  )
        self.spectral_filter_generator = SpectralFilterGeneratorLDA()
        self.seg_model = UnetModel(classes=num_classes,encoder_weights=encoder_weights,in_channels=1)

        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.f1_score = torchmetrics.F1Score(task="multiclass",num_classes=num_classes)
        # Example class weights (replace these with your calculated weights)
        class_weights = torch.tensor([0.2, 1, 5, 80.0, 20.0], dtype=torch.float32)

        if unet_checkpoint:
            # Load the weights from the checkpoint into self.seg_model
            checkpoint = torch.load(unet_checkpoint, map_location=self.device)
            # Adjust the keys
            adjusted_state_dict = {key.replace('seg_model.', ''): value
                                   for key, value in checkpoint['state_dict'].items()}
            # Filter out unexpected keys
            model_keys = set(self.seg_model.state_dict().keys())
            filtered_state_dict = {k: v for k, v in adjusted_state_dict.items() if k in model_keys}

            self.seg_model.load_state_dict(filtered_state_dict, strict=False)

        # If you're using a GPU, remember to transfer weights to the same device as your model
        if torch.cuda.is_available():
            class_weights = class_weights.cuda()
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        self.log_images_every_n_steps = log_images_every_n_steps
        
        self.writer = SummaryWriter(log_dir)


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
        selected_spectra = spectra[:,300:,:]
        selected_spectra_labels = spectra_labels[:,300:]

        self.mean_spectra = calculate_mean_spectra(spectra, spectra_labels)

        spectral_filters = self.spectral_filter_generator.generate_filter(selected_spectra, selected_spectra_labels,n_lda_components=1)
        self.spectral_filters = spectral_filters
        image = self.take_images(spectral_filters, hyperspectral_cube)

        return self.seg_model(image)
    
    def take_images(self, spectral_filters, hyperspectral_cube):

        filter_positive = spectral_filters[0]
        resulting_image1 = self.acquisition_model(filter_positive[:,0,:],hyperspectral_cube)

        filter_negative = spectral_filters[1]
        resulting_image2 = self.acquisition_model(-1*filter_negative[:,0,:],hyperspectral_cube)

        result = resulting_image1 - resulting_image2
        # Convert result to double
        result_float = result.float()
        
        normalized_image = self._normalize_data_by_itself(result_float)

        self.result_float = normalized_image

        return normalized_image

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

    def plot_spectral_filter(self,spectral_filter,target_sinus,mean_spectra):
        spectral_filter1 =spectral_filter[0]
        spectral_filter2 =spectral_filter[1]
        batch_size, nb_spectra_filters,spectrum_len = spectral_filter1.shape
        # Create a figure with subplots arranged horizontally
        fig, axs = plt.subplots(1, batch_size, figsize=(batch_size * 5, 4))  # Adjust figure size as needed

        # Check if batch_size is 1, axs might not be iterable
        if batch_size == 1:
            axs = [axs]

        # Plot each spectral filter in its own subplot
        for i in range(batch_size):

            for j in range(nb_spectra_filters):
                spec_filter1 = spectral_filter1[i,j,:].cpu().detach().numpy()
                spec_filter1 = spec_filter1*0.1 + 0.3
                spec_filter2 = spectral_filter2[i,j,:].cpu().detach().numpy()
                spec_filter2 = spec_filter2*0.1 + 0.3
                axs[i].plot(self.optical_model.wavelengths,spec_filter1,label="spectral filter nb "+str(j),linestyle='--')
                axs[i].plot(self.optical_model.wavelengths,spec_filter2,label="spectral filter nb "+str(j),linestyle='-')
            axs[i].set_title(f"Spectral Filter {i + 1}")
            axs[i].set_xlabel("Wavelength index")
            axs[i].set_ylabel("Filter value")
            axs[i].grid(True)

            for j in range(5):
                mean_spec = mean_spectra[i,j,:].cpu().detach().numpy()
            # diff_norm =  mean_spectra[i,4,:].cpu().detach().numpy() - mean_spectra[i,3,:].cpu().detach().numpy().min()
            # diff = diff_norm / diff_norm.max()
            # diff = diff
                axs[i].plot(self.optical_model.wavelengths,mean_spec,label="spectrum nb "+str(j))
            # axs[i].plot(self.optical_model.wavelengths,diff,label="difference 4 and 5")
            # axs[i].plot(self.optical_model.wavelengths,target_sinus[i,0,:].cpu().detach().numpy(),label="target sinus")
            axs[i].set_ylim(0.25,0.5)
            plt.legend()
        # Adjust layout
        # plt.ylim(0.25,0.5)

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
        #     plt.figure()
        #     plt.imshow(self.result_float[0,0,...].detach().cpu().numpy())
        #     plt.colorbar()
        #     plt.savefig(f"./test_images/{self.global_step}.png")

        if batch_idx % self.log_images_every_n_steps == 0:

            predicted_maps = self._convert_output_to_images(y_hat)
            acquired_images = self._normalize_image_tensor(self.result_float)

            y_one_hot = F.one_hot(y, num_classes=5)
            y_one_hot = y_one_hot.permute(0, 3, 1, 2)
            ground_truth_maps = self._convert_output_to_images(y_one_hot)

            self._log_images('train/predicted_maps', predicted_maps, self.global_step)
            self._log_images('train/acquired_images', acquired_images, self.global_step)
            self._log_images('train/ground_truth_maps', ground_truth_maps, self.global_step)
            # Generating and logging the spectral filter plot
            spectral_filter_plot = self.plot_spectral_filter(self.spectral_filters,None,self.mean_spectra)
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
            self._log_images('val/input_images', input_images, self.global_step)

        return {"loss": loss, "scores":y_hat, "y":y}

    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx)

        SNR_indiv = self.acquisition_model.SNR


        # Convert y_hat to class predictions
        y_hat_pred = torch.argmax(y_hat, dim=1)

        # Metrics for Classes 3 & 4 Combined
        y_hat_classes_34 = torch.where((y_hat_pred == 3) | (y_hat_pred == 4), y_hat_pred,
                                       torch.tensor(0, device=y_hat_pred.device))
        y_classes_34 = torch.where((y == 3) | (y == 4), y, torch.tensor(0, device=y.device))

        # Calculate metrics for classes 3 & 4
        metrics_classes_34 = calculate_metrics(y_hat_classes_34, y_classes_34, label_of_interest=[3, 4])

        # Metrics for All Classes
        metrics_all_classes = calculate_metrics(y_hat_pred, y,
                                                label_of_interest=None)  # Pass None or all class labels to consider all

        self.log_dict(
            {
                "test_loss": loss,
                **{"class_34_" + k: v for k, v in metrics_classes_34.items()},
                **{"overall_" + k: v for k, v in metrics_all_classes.items()},
                "SNR_indiv": SNR_indiv,
            },
            on_step=False,  # Change to True if you want to log each step
            on_epoch=True,
            prog_bar=True,
        )


        if not os.path.exists("./test_images_LDA_PCA_2shots"):
            os.makedirs("./test_images_LDA_PCA_2shots")

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.imshow(self.result_float[0, 0, ...].detach().cpu().numpy(), cmap="gray")
        # ax[1].imshow(self.result_float[0, 1, ...].detach().cpu().numpy())
        # plt.colorbar()
        plt.savefig(f"./test_images_LDA_PCA_2shots/{batch_idx}_image_{self.readout_noise_level}.png")
        plt.close()

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        # diff_norm = self.mean_spectra[0, 4, :].cpu().detach().numpy() - self.mean_spectra[0, 3,
        #                                                                 :].cpu().detach().numpy().min()
        # diff = diff_norm / diff_norm.max()
        # diff = diff
        for i in range(5):
            ax.plot(self.optical_model.wavelengths, self.mean_spectra[0, i, :].cpu().detach().numpy(),
                    label="spectrum nb " + str(i))
        # ax.plot(self.optical_model.wavelengths, diff, label="difference 4 and 5")
        ax.plot(self.spectral_filters[0].detach().cpu().numpy()[0,0,:], label="spectral filter")
        ax.set_xlim(0.7, 0.86)
        # ax[1].imshow(self.result_float[0, 1, ...].detach().cpu().numpy())
        # plt.colorbar()
        plt.savefig(f"./test_images_LDA_PCA_2shots/{batch_idx}_spectra_{self.readout_noise_level}.png")
        plt.close()

        return {"loss": loss, "scores":y_hat, "y":y}

    def predict_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx)
        self.log('predict_step', loss,on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def _common_step(self, batch, batch_idx):
        hyperspectral_cube, spectra,spectra_labels,y,_ = batch
        x = (hyperspectral_cube, spectra,spectra_labels)
        y_hat = self.forward(x)
        y = y.long()

        loss = self.loss_fn(y_hat, y)
        return loss, y_hat, y

    def on_test_end(self):
        metrics = self.trainer.logged_metrics  # This contains all logged metrics
        # Convert metrics to a DataFrame or any format you prefer
        metrics = {k: float(v) for k, v in metrics.items()}
        # Convert metrics to a DataFrame or any format you prefer
        metrics_df = pd.DataFrame([metrics])
        # File path
        file_path = './test_results_LDA_PCA_2_acq.csv'
        # Check if file exists to decide whether to write header
        file_exists = os.path.isfile(file_path)
        # Save to CSV, append if file exists, include header if file does not exist
        metrics_df.to_csv(file_path, mode='a', index=False, header=not file_exists)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    
