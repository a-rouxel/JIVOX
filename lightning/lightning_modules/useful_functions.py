import torch




def calculate_mean_spectra(spectra_array, labels):
    batch_size, _, spectrum_len = spectra_array.shape
    max_labels_per_batch = 5  # Adjust based on your needs
    mean_spectra_list = []

    # Process each batch
    for i in range(batch_size):
        batch_spectra = spectra_array[i]  # Shape: (500, spectrum_len)
        batch_labels = labels[i]  # Shape: (500,)
        unique_labels = torch.unique(batch_labels)
        batch_mean_spectra = []

        # Compute mean spectra for up to 5 unique labels
        for label in unique_labels[:max_labels_per_batch]:  # Limiting to first N labels
            mask = batch_labels == label
            mean_spectrum = batch_spectra[mask].mean(dim=0)
            batch_mean_spectra.append(mean_spectrum)

        # Ensure each batch has N mean spectra
        while len(batch_mean_spectra) < max_labels_per_batch:
            # Append zeros or handle as needed
            batch_mean_spectra.append(torch.zeros(spectrum_len, dtype=batch_spectra.dtype, device=batch_spectra.device))

        # Stack mean spectra for the current batch
        batch_mean_spectra_tensor = torch.stack(batch_mean_spectra)
        mean_spectra_list.append(batch_mean_spectra_tensor)

    # Stack all batches together
    mean_spectra = torch.stack(mean_spectra_list).float()  # Shape: (batch_size, max_labels_per_batch, spectrum_len)

    return mean_spectra

