import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
import random
import torch



class AcquisitionModel(pl.LightningModule):

    def __init__(self, input_spectrum,optical_model,
                 psf_width=0.9,
                 auto_exposure="False",
                 random_readout_noise="False",
                 readout_noise_value=0.005):
        super().__init__()
        self.readout_noise_value = readout_noise_value
        self.psf_width = psf_width
        self.optical_model = optical_model
        self.auto_exposure = auto_exposure
        self.random_readout_noise = random_readout_noise
        self.register_buffer('input_spectrum', input_spectrum)
        self.register_buffer('spatial_psf', torch.from_numpy(self.optical_model.spatial_psf))

        psf = self.generate_spectral_psf()
        self.register_buffer('psf', psf)

    def generate_spectral_psf(self):
        # generate wavelengths vector
        wavelengths = torch.linspace(-5, 5, 11)
        # generate psf
        psf = torch.exp(-wavelengths ** 2 / (2*self.psf_width))
        # normalize psf
        psf = psf / torch.sum(psf)
        psf = psf.unsqueeze(0)
        psf = psf.repeat(1, 1,1)

        return psf

    def high_order_gaussian(self,size, sigma, order=2):
        """
        Create a high-order Gaussian filter in PyTorch.
        size: the size of the filter (height, width).
        sigma: standard deviation of the Gaussian.
        order: the power to which the Gaussian is raised.
        """
        Y, X = torch.meshgrid(torch.arange(size[0],device=self.device), torch.arange(size[1],device=self.device), indexing='ij')
        y, x = size[0] // 2, size[1] // 2
        gaussian = torch.exp(-((X - x) ** 2 + (Y - y) ** 2) / (2. * sigma ** 2))

        return gaussian ** order

    def generate_newtons_rings_pattern(self,intensity_factor, size):
        """
        Generate a Newton's Rings pattern in PyTorch.
        size: size of the pattern in pixels (width, height)
        intensity_factor: controls the intensity of the pattern
        """
        size = torch.from_numpy(size).to(self.device)
        wavelength = size[0] / 2
        Y, X = torch.meshgrid(torch.arange(size[1],device=self.device), torch.arange(size[0],device=self.device), indexing='ij')
        center = torch.tensor([size[1] // 2, size[0] // 2],device=self.device)
        R = torch.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
        intensity = torch.sin((R ** 2) / wavelength)

        high_order = self.high_order_gaussian(size, sigma=size[0] // 2, order=8)

        result = intensity_factor * intensity * high_order

        return result

    def convolve_spectral_filter_with_psf(self,spectral_filter):

        # try:
        #     spectral_filter = spectral_filter.unsqueeze(1)
        # except Exception as e:
        #     pass

        if spectral_filter.shape[0] == 1:
            convolved_spectral_filter = F.conv1d(spectral_filter, self.psf, padding='same').squeeze(0)
        else:
            try:
                convolved_spectral_filter = F.conv1d(spectral_filter, self.psf, padding='same').squeeze()
            except:
                spectral_filter = spectral_filter.unsqueeze(1)
                convolved_spectral_filter = F.conv1d(spectral_filter, self.psf, padding='same').squeeze()

        return convolved_spectral_filter
    


    def filter_input_spectrum(self,input_spectrum,filter):
        # Element-wise multiplication with input_spectrum
        # Ensure that input_spectrum is correctly shaped for element-wise multiplication
        filtered_input_spectrum = filter * input_spectrum

        return filtered_input_spectrum

    def filter_cube_with_spectral_shaper(self, scene_hyperspectral_cube,spectral_shaper_filter):

        # Reshape filtered_input_spectrum to make it [1, 1, 1, wavelengths]
        # This is necessary for proper broadcasting across the batch and spatial dimensions
        # print(spectral_shaper_filter.dim())
        if spectral_shaper_filter.dim() != 2:
            spectral_shaper_filter = spectral_shaper_filter.unsqueeze(0)
        spectral_shaper_filter = spectral_shaper_filter.unsqueeze(1).unsqueeze(1)
        # print(spectral_shaper_filter.shape)
        # print(scene_hyperspectral_cube.shape)
        filtered_hyperspectral_cube = scene_hyperspectral_cube * spectral_shaper_filter

        # No need to reshape it back as it retains its original shape due to broadcasting
        return filtered_hyperspectral_cube

    def apply_qe(self, filtered_hyperspectral_cube, qe):
        # multiply the filtered hyperspectral cube with the quantum efficiency in the wavelength dimension
        qe = torch.from_numpy(qe).to(self.device).float()
        qe = qe.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        qe = qe.repeat(filtered_hyperspectral_cube.shape[0],1,1,1)
        filtered_hyperspectral_cube = filtered_hyperspectral_cube * qe
        return filtered_hyperspectral_cube
    
    def integrate_along_wavelength(self, filtered_hyperspectral_cube):
        # Integrate along wavelength dimension
        integrated_hyperspectral_cube = torch.sum(filtered_hyperspectral_cube, dim=-1)

        return integrated_hyperspectral_cube

    def apply_spatial_psf(self, image):
        # Calculate the padding size
        # Assuming self.spatial_psf is a square kernel for simplicity
        kernel_height, kernel_width = self.spatial_psf.shape[-2], self.spatial_psf.shape[-1]
        pad_height = (kernel_height - 1) // 2
        pad_width = (kernel_width - 1) // 2

        # Pad the image with replication of edge values
        padded_image = F.pad(image, (pad_width, pad_width, pad_height, pad_height), mode='replicate')


        # Add a channel dimension to your images
        padded_image = padded_image.unsqueeze(1)  # New shape: [2, 1, 520, 520]

        # Reshape PSF to include channel dimension and make it a single filter
        spatial_psf = self.spatial_psf.unsqueeze(0).unsqueeze(0)  # New shape: [1, 1, 9, 9]

        blurred_image = F.conv2d(padded_image, spatial_psf)

        return blurred_image

    def add_shot_noise(self, integrated_hyperspectral_cube, factor=0.02):
        """
        Add shot noise (Poisson noise) to the integrated hyperspectral cube.
        factor: A scaling factor for adjusting the noise level.
        """
        # Ensure non-negative values for Poisson noise application
        lambda_values = torch.clamp(integrated_hyperspectral_cube, min=0) * factor

        # Debugging: Check min value
        # print("Min lambda value before Poisson:", lambda_values.min().item())

        # Ensure strictly non-negative values to avoid floating-point issues
        lambda_values = torch.clamp(lambda_values, min=0)

        # Apply Poisson noise
        # The Poisson distribution inherently scales the noise with the square root of the signal intensity
        shot_noise = torch.poisson(lambda_values) - lambda_values

        # Rescale and add the noise to the original cube
        noisy_integrated_hyperspectral_cube = integrated_hyperspectral_cube + shot_noise * factor
        return noisy_integrated_hyperspectral_cube

    def add_readout_noise(self, integrated_hyperspectral_cube, mu=0.0, sigma=1.0):
        """
        Add readout noise (Gaussian noise) to the integrated hyperspectral cube.
        mu: Mean of the Gaussian readout noise
        sigma: Standard deviation of the Gaussian readout noise
        """
        readout_noise = torch.normal(mu, sigma, size=integrated_hyperspectral_cube.shape,
                                     device=integrated_hyperspectral_cube.device)
        noisy_integrated_hyperspectral_cube = integrated_hyperspectral_cube + readout_noise

        return noisy_integrated_hyperspectral_cube

    def add_newton_rings(self, integrated_hyperspectral_cube,num_patterns=3, size_range=(8, 50)):
        """
        Overlay multiple Newton's Rings patterns of various sizes on the integrated hyperspectral cube.
        num_patterns: Number of Newton's Rings patterns to add
        size_range: Tuple of (min_size, max_size) for the rings
        """
        cube_height, cube_width = integrated_hyperspectral_cube.shape[1], integrated_hyperspectral_cube.shape[2]
        updated_hyperspectral_cube = integrated_hyperspectral_cube.clone()
        mean_intensity = torch.mean(integrated_hyperspectral_cube, dim=(1, 2), keepdim=True)

        for _ in range(num_patterns):
            # Randomly determine the size of the current pattern
            pattern_size = random.randint(size_range[0], size_range[1])
            patterns = np.array([pattern_size, pattern_size])
            intensity_factor = random.uniform(0.005, 0.1)
            # Generate Newton's Rings pattern
            newton_rings_pattern = self.generate_newtons_rings_pattern(intensity_factor, patterns)

            normalized_newton_rings_pattern = newton_rings_pattern
            # normalized_newton_rings_pattern = newton_rings_pattern
            # Randomly determine position to place the pattern
            x_offset = random.randint(0, cube_width - pattern_size)
            y_offset = random.randint(0, cube_height - pattern_size)

            # Overlay the pattern onto the cube
            updated_hyperspectral_cube[:, y_offset:y_offset + pattern_size,
            x_offset:x_offset + pattern_size] += normalized_newton_rings_pattern

        return updated_hyperspectral_cube

    def forward(self, spectral_filter, scene_hyperspectral_cube,seed_value=0):

        if seed_value ==0:
            seed_value = random.randint(0, 1000)
            local_rng = random.Random()
            local_rng.seed(seed_value)
        else:
            local_rng = random.Random()
            local_rng.seed(seed_value)


        #convolve spectral filter with spectral psf
        convolved_spectral_filter = self.convolve_spectral_filter_with_psf(spectral_filter)

        # Filter input spectrum with spectral filter
        filtered_input_spectrum = self.filter_input_spectrum(self.input_spectrum, 
                                                             convolved_spectral_filter)

        # Filter hyperspectral cube with spectral shaper
        filtered_hyperspectral_cube = self.filter_cube_with_spectral_shaper(scene_hyperspectral_cube,
                                                                            filtered_input_spectrum)

        # apply QE of the detector
        filtered_hyperspectral_cube = self.apply_qe(filtered_hyperspectral_cube,
                                                    self.optical_model.quantum_efficiency)

        # Integrate along wavelength dimension
        integrated_hyperspectral_cube = self.integrate_along_wavelength(filtered_hyperspectral_cube)
        #

        mean_intensity = torch.mean(integrated_hyperspectral_cube, dim=(1, 2), keepdim=True)
        mean_intensity[mean_intensity < 1e-6] = 1

        if self.auto_exposure:
            # get mean intensity

            integrated_hyperspectral_cube = integrated_hyperspectral_cube * 1/mean_intensity
            readout_noise_value = self.readout_noise_value
            rand_readout_noise = local_rng.uniform(readout_noise_value/3, readout_noise_value*10)
        else:
            readout_noise_value = self.readout_noise_value
            rand_readout_noise = local_rng.uniform(readout_noise_value/10, readout_noise_value*10)

        # add newton rings
        integrated_hyperspectral_cube = self.add_newton_rings(integrated_hyperspectral_cube,num_patterns=random.randint(1, 30),
                                                              size_range=(2, 15))
        # apply spatial psf
        integrated_hyperspectral_cube = self.apply_spatial_psf(integrated_hyperspectral_cube)


        # add noise
        noisy_integrated_hyperspectral_cube = self.add_shot_noise(integrated_hyperspectral_cube,factor=0.005)

        # clip negative values
        # noisy_integrated_hyperspectral_cube[noisy_integrated_hyperspectral_cube < 0] = 0

        if self.random_readout_noise:
            noisy_integrated_hyperspectral_cube = self.add_readout_noise(noisy_integrated_hyperspectral_cube,
                                                                         mu=0.1,
                                                                         sigma=rand_readout_noise)
        else:
            noisy_integrated_hyperspectral_cube = self.add_readout_noise(noisy_integrated_hyperspectral_cube,
                                                                         mu=0.1,
                                                                         sigma=readout_noise_value)
        
        self.SNR = 20*torch.log10(torch.mean(integrated_hyperspectral_cube)/readout_noise_value)

        noisy_integrated_hyperspectral_cube = torch.clamp(noisy_integrated_hyperspectral_cube, min=0)

        return noisy_integrated_hyperspectral_cube
