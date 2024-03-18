import numpy as np
mm = 10e-3

def gaussian_1D(X_pupil_space,waist=2*mm):

    gaussian = np.exp(-1*X_pupil_space**2/(waist**2))

    return gaussian

def apply_obj_entrance_pupil_cropping(X_pupil_space, beam_profile_input_obj,entrance_pupil_diameter):

    beam_profile_input_obj[np.abs(X_pupil_space)>entrance_pupil_diameter/2] = 0

    return beam_profile_input_obj

def position_to_angle(position, focal_length):
    return np.arctan(position/focal_length)




class OpticalModel:
    def __init__(self,spectrum=None):
        self.NA = 0.4
        self.f_obj = 10*mm # carac. Mitutoyo x20
        self.waist = 2*mm # waist of the input_amplitude in the pupil plane
        self.waist_image = 0.002*mm # waist of the output_amplitude in the image plane
        self.X_pupil_space = np.linspace(-8 * mm, 8 * mm, 10)
        self.X_image_space = np.linspace(-0.005 * mm, 0.005 * mm, 9)
        self.wavelengths = np.linspace(0.7, 0.86, 100)
        self.quantum_efficiency = self.define_quantum_efficiency()
        self.num_wavelengths = self.wavelengths.shape[0]

        self.obj_entrance_pupil_diameter = self.NA*2*self.f_obj
        self.spectrum = spectrum

        self.amplitude_distribution_pupil_plane = self.generate_amplitude_distribution_pupil_plane(self.waist)
        self.angles = self.generate_angles_on_sample()

        self.spatial_psf = self.generate_spatial_psf()

    def define_quantum_efficiency(self,camera_type="BASLER-VIS"):
        if camera_type == "BASLER-VIS":
            wavelengths = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
            quantum_efficiency = [32,54,62,64,61,54,45,35,24,18,12,8,2]

            # interpolation of the quantum efficiency to the wavelengths
            quantum_efficiency = np.interp(self.wavelengths,wavelengths,quantum_efficiency)
            quantum_efficiency *= 0.01

        else:
            raise ValueError("Camera type not recognized")

        return quantum_efficiency


    def generate_spatial_psf(self):
        X,Y = np.meshgrid(self.X_image_space,self.X_image_space)
        spatial_psf = np.exp(-1*(X**2+Y**2)/(self.waist_image**2))
        #normalize psf so the sum is equal to one
        spatial_psf = spatial_psf/np.sum(spatial_psf)

        return spatial_psf
    def generate_amplitude_distribution_pupil_plane(self,waist):
        amplitude_distribution_pupil_plane = gaussian_1D(self.X_pupil_space,waist=waist)
        return amplitude_distribution_pupil_plane

    def propagate_through_objective_lens(self):
        beam_profile_input_obj_cropped = apply_obj_entrance_pupil_cropping(self.X_pupil_space, self.amplitude_distribution_pupil_plane,
                                                                           self.obj_entrance_pupil_diameter)
        return beam_profile_input_obj_cropped

    def generate_angles_on_sample(self):
        angles = position_to_angle(self.X_pupil_space,self.f_obj)
        return angles


if __name__ == "__main__":
    optical_model = OpticalModel()

