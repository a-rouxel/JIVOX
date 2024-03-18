from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import logging
import numpy as np
import torch

# Define a simple moving average function
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

def gaussian_psf(delta_lambda, window_size):
    """ Creates a Gaussian Point Spread Function (PSF).

    Args:
    delta_lambda (float): The standard deviation of the Gaussian, controlling the smoothing.
    window_size (int): The number of points in the Gaussian window.

    Returns:
    numpy.ndarray: The Gaussian PSF.
    """
    sigma = delta_lambda
    x = np.linspace(-window_size // 2, window_size // 2, window_size)
    psf = np.exp(-x**2 / (2 * sigma**2))
    return psf / psf.sum()  # Normalize the PSF

def smooth_curve_with_psf(curve, delta_lambda, window_size):
    """ Smooths a curve by convolving it with a Gaussian PSF.

    Args:
    curve (numpy.ndarray): The curve to be smoothed.
    delta_lambda (float): The standard deviation of the Gaussian PSF.
    window_size (int): The number of points in the Gaussian window.

    Returns:
    numpy.ndarray: The smoothed curve.
    """
    psf = gaussian_psf(delta_lambda, window_size)
    smoothed_curve = np.convolve(curve, psf, mode='same')
    return smoothed_curve

def apply_pca_and_lda(data, labels, n_pca_components=5, n_lda_components=1, delta_lambda=10, window_size=50):
    """
    Applies PCA and LDA on the data, maps LDA coefficients back to the original feature space,
    and smooths the coefficients using a Gaussian PSF.

    Args:
    data (numpy.ndarray): The input data.
    labels (numpy.ndarray): The labels for the input data.
    n_pca_components (int): Number of components for PCA.
    n_lda_components (int): Number of components for LDA.
    delta_lambda (float): Standard deviation for the Gaussian PSF.
    window_size (int): Window size for the Gaussian PSF.

    Returns:
    numpy.ndarray: Normalized LDA coefficients mapped back to the original feature space.
    """
    # Apply PCA
    pca = PCA(n_components=n_pca_components)

    normalized_lda_coef_3D = torch.zeros((data.shape[0],n_lda_components, data.shape[2]))
    lda_eigenvalues_3D = torch.zeros((data.shape[0], n_lda_components))

    for i in range(data.shape[0]):

        X_pca = pca.fit_transform(data[i,:,:])

        # Apply LDA on PCA-reduced data
        lda = LDA(n_components=n_lda_components)
        X_lda = lda.fit_transform(X_pca, labels[i,:])

        # Compute the covariance matrix of the PCA-reduced data
        cov_matrix = np.cov(X_pca.T)

        # Compute the eigenvalues for the LDA components
        lda_eigenvalues = np.diag(np.dot(np.dot(lda.scalings_.T, cov_matrix), lda.scalings_))
        # PCA loadings (mapping of original features to PCA components)
        pca_loadings = pca.components_

        for j in range(n_lda_components):

            # LDA coefficients in the PCA-reduced space
            lda_coef_in_pca_space = lda.scalings_[:,j].ravel()


            # Map LDA coefficients back to the original feature space
            lda_coef_in_original_space = np.dot(pca_loadings.T, lda_coef_in_pca_space)

            # Smooth the LDA coefficients using Gaussian PSF
            # smoothed_lda_coef = smooth_curve_with_psf(lda_coef_in_original_space, delta_lambda, window_size)
            smoothed_lda_coef = lda_coef_in_original_space

            # Normalize the smoothed coefficients
            normalized_lda_coef = smoothed_lda_coef / max(np.abs(smoothed_lda_coef))

            # normalized_lda_coef *= 0.02

            normalized_lda_coef_3D[i,j,:] = torch.from_numpy(normalized_lda_coef)

        # lda_eigenvalues_3D[i,:] = torch.from_numpy(lda_eigenvalues)
        logging.info(f"PCA and LDA applied with n_pca_components={n_pca_components}, n_lda_components={n_lda_components}")

    return normalized_lda_coef_3D, None

def generate_spectral_filters(normalized_lda_coef_in_original_space=None,filter_generation_type="LDA"):
    """ Generates spectral filters.

    Args:
    filter_generation_type (str): The type of filter to be generated. Can be "LDA" or "PCA".
    normalized_lda_coef_in_original_space (numpy.ndarray): The normalized LDA coefficients mapped back to the
    original feature space.

    Returns:
    numpy.ndarray: The spectral filters.
    """
    spectral_filters = list()
    if filter_generation_type == "LDA":
        # Initialize filters as copies
        spectral_filter1 = np.copy(normalized_lda_coef_in_original_space)
        spectral_filter2 = np.copy(normalized_lda_coef_in_original_space)

        # Set values to zero that don't meet the condition
        # For spectral_filter1, set all values <= 0 to zero
        spectral_filter1[spectral_filter1 <= 0] = 0

        # For spectral_filter2, set all values >= 0 to zero
        spectral_filter2[spectral_filter2 >= 0] = 0

        # Append filters to the spectral_filters list
        spectral_filters.append(spectral_filter1)
        spectral_filters.append(spectral_filter2)
    else:
        raise ValueError("filter_generation_type unknown")


    return spectral_filters