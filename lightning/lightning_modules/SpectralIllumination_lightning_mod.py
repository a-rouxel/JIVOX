import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.functional import accuracy
from SpectralFilterGenerator import ResNet1D_Peter,ResNet1D_Peter_with_optim
import matplotlib.pyplot as plt
import numpy as np
import os

class ResNet1DLightning(pl.LightningModule):
    def __init__(self, optical_model, learning_rate=5e-4):

        self.optical_model = optical_model

        super(ResNet1DLightning, self).__init__()
        # Your existing ResNet1D architecture
        self.model = ResNet1D_Peter(input_channels=5,
                                    wavelengths_vec=optical_model.wavelengths,num_classes=2)

        # #load existing checkpoint
        # checkpoint = torch.load(
        #     os.path.join(os.getcwd(), "tb_logs/training_sinusoide/version_24/checkpoints/epoch=82-step=16600.ckpt"))
        # self.model.load_state_dict(torch.load(os.path.join(os.getcwd(), "./tb_logs/training_sinusoide/version_24/checkpoints/epoch=82-step=16600.ckpt")['state_dict']))

        self.learning_rate = learning_rate
        self.loss_fn = nn.L1Loss()  # or nn.MSELoss() for regression
        
        # For saving hyperparameters and enabling easy model checkpoints and reloads
        self.save_hyperparameters()

    def preprocess_spectra_array(self, spectra_array):

        mean = torch.mean(spectra_array, dim=2, keepdim=True)
        std_dev = torch.std(spectra_array, dim=2, keepdim=True)
        std_dev[std_dev == 0] = 1e-10

        # Standardize the data
        normalized_data = (spectra_array - mean) / std_dev

        return normalized_data

    def forward(self, x):
        # Forward pass through the model defined in ResNet1D
        last_spectra = x
        last_spectra = last_spectra.float()

        diff = last_spectra[:,1,:] - last_spectra[:,0,:]
        diff = diff.unsqueeze(1)


        # normalized_spectra = self.preprocess_spectra_array(last_spectra)
        # normalized_spectra = diff
        # diff = diff / torch.max(torch.abs(diff))

        self.last_spectra = last_spectra
        diff = diff / torch.abs(torch.max(diff))
        self.diff = diff
        # plt.plot(normalized_spectra[0,0,:].detach().numpy())
        # plt.plot(normalized_spectra[0,1,:].detach().numpy())
        # plt.show()


        x = self.model(last_spectra )

        # print("x grad ", x.grad)

        # print(x.shape)
        # print(x)
        f_max = 30
        f_min = 12
        phase_min =0
        phase_max = 2*torch.pi


        f_new = (f_max - f_min) / 2 * x[:, 0] + (f_max + f_min) / 2
        phase_new = (phase_max - phase_min) / 2 * x[:, 1] + (phase_max + phase_min) / 2
        x_modified = torch.stack((f_new, phase_new), dim=1)

        x_modified = x_modified.unsqueeze(1)
        generated_sinus = self.model.generate_sinusoids(x_modified)


        return generated_sinus, x_modified

    def configure_optimizers(self):
        # Configure your optimizer and optionally your LR scheduler
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        # Training step
        x, y = batch
        y_hat, y_before = self(x)

        y = y.unsqueeze(1)

        print(y_before)

        target_sinus = self.model.generate_sinusoids(y)

        loss1 = 20*self.loss_fn(y_hat, target_sinus)


        # plt.plot(target_sinus[0, 0, :].cpu().detach().numpy(), label="target")
        # plt.plot(y_hat[0, 0, :].cpu().detach().numpy(), label="generated")
        # plt.show()
        loss_freq = self.loss_fn(y_before[:,:,0], y[:,:,0])
        loss_phase = self.loss_fn(y_before[:,:,1], y[:,:,1])
        loss2 = loss_freq**2


        loss =  loss2 + loss1


        # print(y_hat.shape)
        # print(target_sinus.shape)

        # loss = torch.mean(torch.sum((y_hat[:,0,:]- target_sinus[:,0,:])**2))

        # print(loss)

        # Example: Print gradients of the first layer's weights in your model
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.grad)
        #         break  # Remove this if you want to see all gradients

        self.log_dict(
            { "train_loss": loss,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )


        return loss

    def validation_step(self, batch, batch_idx):
        # Validation step
        x, y = batch
        y_hat, y_before = self(x)

        y = y.unsqueeze(1)
        # y_hat = y_hat.unsqueeze(1)

        target_sinus = self.model.generate_sinusoids(y)
        # generated_sinus = self.model.generate_sinusoids(y_hat)




        # val_loss = self.loss_fn(y_hat, target_sinus)
        loss1 = 20*self.loss_fn(y_hat, target_sinus)


        # plt.plot(target_sinus[0, 0, :].cpu().detach().numpy(), label="target")
        # plt.plot(y_hat[0, 0, :].cpu().detach().numpy(), label="generated")
        # plt.show()
        loss_freq = self.loss_fn(y_before[:,:,0], y[:,:,0])
        loss_phase = self.loss_fn(y_before[:,:,1], y[:,:,1])
        loss2 = loss_freq**2


        val_loss =  loss2 + loss1
        # val_loss = torch.mean(torch.sum((y_hat[:,0,:]- target_sinus[:,0,:])**2))

        # val_loss = self.loss_fn(y_hat, y)
        self.log('val_loss', val_loss)

        # Save the first 5 spectra for the first batch of each epoch
        if batch_idx == 0:  # Check if it's the beginning of the epoch
            print("here we GOOOOOOOOOOOOo")
            save_dir = "validation_plots"
            os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
            epoch = self.current_epoch  # Get the current epoch number

            for i in range(5):  # Assuming y and y_hat have at least 5 elements in the batch dimension
                plt.figure(figsize=(10, 6))
                plt.plot(target_sinus[i, 0, :].cpu().detach().numpy(), label="Target")
                plt.plot(y_hat[i, 0, :].cpu().detach().numpy(), label="Generated")
                # If you want to plot the inputs and their difference as well
                # plt.plot(self.last_spectra[i, 0, :].cpu().detach().numpy(), label="Input1")
                # plt.plot(self.last_spectra[i, 1, :].cpu().detach().numpy(), label="Input2")
                plt.plot(self.diff[i, 0, :].cpu().detach().numpy(), label="Diff")
                plt.title(f"Spectra Comparison for Sample {i + 1}")
                plt.legend()
                plt.xlabel("Wavelength")
                plt.ylabel("Intensity")
                plot_path = os.path.join(save_dir, f"epoch_{epoch}_sample_{i + 1}.png")
                plt.savefig(plot_path)
                plt.close()  # Close the plot to free memory


class ResNet1DLightning_optuna(pl.LightningModule):
    def __init__(self, optical_model,
                        learning_rate=1e-3,
                        num_blocks_per_layer=[3, 3, 2, 2],
                        channels_per_block=[32, 64, 128, 256],
                        common_dense_size=256):

        self.optical_model = optical_model

        super(ResNet1DLightning_optuna, self).__init__()
        # Your existing ResNet1D architecture
        self.model = ResNet1D_Peter_with_optim(input_channels=5,
                                               wavelengths_vec=optical_model.wavelengths,
                                               num_classes=2,
                                               num_blocks_per_layer=num_blocks_per_layer,
                                               channels_per_block=channels_per_block,
                                               common_dense_size=common_dense_size)

        self.learning_rate = learning_rate
        self.loss_fn = nn.L1Loss()  # or nn.MSELoss() for regression

        # For saving hyperparameters and enabling easy model checkpoints and reloads
        self.save_hyperparameters()

    def preprocess_spectra_array(self, spectra_array):

        mean = torch.mean(spectra_array, dim=2, keepdim=True)
        std_dev = torch.std(spectra_array, dim=2, keepdim=True)
        std_dev[std_dev == 0] = 1e-10

        # Standardize the data
        normalized_data = (spectra_array - mean) / std_dev

        return normalized_data

    def forward(self, x):
        # Forward pass through the model defined in ResNet1D
        last_spectra = x
        last_spectra = last_spectra.float()

        diff = last_spectra[:, 1, :] - last_spectra[:, 0, :]
        diff = diff.unsqueeze(1)

        # normalized_spectra = self.preprocess_spectra_array(last_spectra)
        # normalized_spectra = diff
        # diff = diff / torch.max(torch.abs(diff))

        self.last_spectra = last_spectra
        diff = diff / torch.abs(torch.max(diff))
        self.diff = diff
        # plt.plot(normalized_spectra[0,0,:].detach().numpy())
        # plt.plot(normalized_spectra[0,1,:].detach().numpy())
        # plt.show()

        print(last_spectra.shape)

        print("normalized_spectra ", last_spectra.shape)
        x = self.model(last_spectra)

        # print("x grad ", x.grad)

        # print(x.shape)
        # print(x)
        f_max = 30
        f_min = 12
        phase_min = 0
        phase_max = 2 * torch.pi

        f_new = (f_max - f_min) / 2 * x[:, 0] + (f_max + f_min) / 2
        phase_new = (phase_max - phase_min) / 2 * x[:, 1] + (phase_max + phase_min) / 2
        x_modified = torch.stack((f_new, phase_new), dim=1)

        x_modified = x_modified.unsqueeze(1)
        generated_sinus = self.model.generate_sinusoids(x_modified)

        print(x[0])
        print(x[1])

        return generated_sinus, x_modified

    def configure_optimizers(self):
        # Configure your optimizer and optionally your LR scheduler
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        # Training step
        x, y = batch
        y_hat, y_before = self(x)

        y = y.unsqueeze(1)

        target_sinus = self.model.generate_sinusoids(y)

        loss1 = 20 * self.loss_fn(y_hat, target_sinus)

        # plt.plot(target_sinus[0, 0, :].cpu().detach().numpy(), label="target")
        # plt.plot(y_hat[0, 0, :].cpu().detach().numpy(), label="generated")
        # plt.show()
        # print(y_before)
        # print("results", y)

        loss_freq = self.loss_fn(y_before[:, :, 0], y[:, :, 0])
        loss_phase = self.loss_fn(y_before[:, :, 1], y[:, :, 1])
        loss2 = loss_freq ** 2

        loss = loss2 + loss1

        # print(y_hat.shape)
        # print(target_sinus.shape)

        # loss = torch.mean(torch.sum((y_hat[:,0,:]- target_sinus[:,0,:])**2))

        # print(loss)

        # Example: Print gradients of the first layer's weights in your model
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.grad)
        #         break  # Remove this if you want to see all gradients

        self.log_dict(
            {"train_loss": loss,
             },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        # Validation step
        x, y = batch
        y_hat, y_before = self(x)

        y = y.unsqueeze(1)
        # y_hat = y_hat.unsqueeze(1)

        target_sinus = self.model.generate_sinusoids(y)
        # generated_sinus = self.model.generate_sinusoids(y_hat)

        # val_loss = self.loss_fn(y_hat, target_sinus)
        loss1 = 20 * self.loss_fn(y_hat, target_sinus)

        # plt.plot(target_sinus[0, 0, :].cpu().detach().numpy(), label="target")
        # plt.plot(y_hat[0, 0, :].cpu().detach().numpy(), label="generated")
        # plt.show()
        loss_freq = self.loss_fn(y_before[:, :, 0], y[:, :, 0])
        loss_phase = self.loss_fn(y_before[:, :, 1], y[:, :, 1])
        loss2 = loss_freq ** 2

        val_loss = loss2 + loss1
        # val_loss = torch.mean(torch.sum((y_hat[:,0,:]- target_sinus[:,0,:])**2))

        # val_loss = self.loss_fn(y_hat, y)
        self.log('val_loss', val_loss)

    def test_step(self, batch, batch_idx):
        # Validation step
        x, y = batch
        y_hat, y_before = self(x)

        y = y.unsqueeze(1)
        # y_hat = y_hat.unsqueeze(1)

        target_sinus = self.model.generate_sinusoids(y)
        # generated_sinus = self.model.generate_sinusoids(y_hat)

        # val_loss = self.loss_fn(y_hat, target_sinus)
        loss1 = 20 * self.loss_fn(y_hat, target_sinus)

        # plt.plot(target_sinus[0, 0, :].cpu().detach().numpy(), label="target")
        # plt.plot(y_hat[0, 0, :].cpu().detach().numpy(), label="generated")
        # plt.show()
        loss_freq = self.loss_fn(y_before[:, :, 0], y[:, :, 0])
        loss_phase = self.loss_fn(y_before[:, :, 1], y[:, :, 1])
        loss2 = loss_freq ** 2

        val_loss = loss2 + loss1
        # val_loss = torch.mean(torch.sum((y_hat[:,0,:]- target_sinus[:,0,:])**2))

        # val_loss = self.loss_fn(y_hat, y)
        self.log('test_loss', val_loss)


