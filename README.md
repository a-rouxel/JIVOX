# JIVOX : Joint Illumination and Inspection of VCSEL OXidation
This repository contains the code used to generate the results submitted to the Optica Imaging Congress 2024 in the article "Computational inspection of VCSEL oxidation
exploiting a spectrally-shaped illuminator", A. Rouxel, S. Calvez, A. Monmayrant, G. Almuneau

## 1. Project Description 

![Alt text](figure.png?raw=true "Title")


### Key features

* Generation of VCSEL-like shapes (circle, hexagon, square) using Blender
* Calculation of various multi-layers structures using Scattering-Matrix algorithm [1], [2]
* Lateral wet oxidation simulation of these structures using the method described in [3]
* Generation of spectral filters with arbitrary spectral shapes or sinusoides only
* Monitoring system modeling (optical setup + detection)
* Segmentation algorithm (Resnet34)


## 2. Installation
1. Clone the repository to your local machine.
```bash
git clone https://github.com/a-rouxel/JIVOX.git
```

2. Install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## 3. Generation of the dataset

## Option A - Download the dataset

The dataset (train and test) is available at the following link : [Link to dataset](https://drive.google.com/drive/folders/1
Put them in a dataset directory at the root of the project.

## Option B - Generate the dataset

All the scripts used to generate the dataset are located in a "data_generation" module

### 3.1. Generation of VCSEL-like shapes using Blender

All VCSEL-like shapes related scripts are located in the "shapes" module 

1. Generate json config files using : generate_vcsel_config_files_blend.py. Please look at the python file comments for more details on the parameters.
```bash
python 01_generate_vcsel_config_files_blend.py
```
These config json files will be stored in the directory specified in "dir_dataset".

2. Run generation of VCSEL-like shapes using : generate_shapes.blend (Blender 3.x) and associated script "python_generation_shapes" (already loaded in the blender file). It will load the configs files generated at the previous step and produce various mask images corresponding to the VCSEL areas


### 3.2. Calculation of various multi-layers structures using s-matrix algorithm

1. Define the multilayer stacks of interest in a json file (for now, there are in "multilayer_stacks")

```bash
python 02_init_generate_spectra_config_files.py
```

2. Calculate the reflectivity of these stacks using the s-matrix algorithm using the implementation described in (https://gitlab.laas.fr/arouxel/s-algorithm)

```bash
python 02_real_generating_spectra_from_stack.py
```

The generated spectra will be stored in the directory specified in "output_dir".


### 3.3. Lateral wet oxidation simulation of these structures 

1. Define the oxidation parameters in the python file "03_oxidyze_aperture.py" (oxidation speed depending on Al concentration, nb of vcsels to oxidize , ...)
2. Run the oxidation simulation using : 
```bash
python 03_oxidyze_aperture.py
```
The resulting images will be stored in the directory specified in "dir_dataset".


### 3.4 Combining previous outputs to generate generic ground truth dataset

1. Generate resulting png files from various masks (corresponding to various steps of the oxidation)
2. Associate the spectra from the multi-layers stacks to the corresponding files
3. Generate the ground truth gif

```bash
python generate_ground_truth_data.py
```

## 4 - Train the (sinus generator +) segmentation model 

1. First, train the segmentation model using an arbitrary spectrum illuminator

```bash
python training_segmentation_LDA_illumination.py
```

2. Pre-train the sinusoide generator using the following command

```bash
python training_sinus_generator.py
```

3. Train the segmentation model using the sinusoidal illuminator

```bash
python segmentation_and_sinus.py
```

## 5 - Testing models

Checkpoints of the models are available in at the following link : [Link to checkpoints](https://cloud.laas.fr/index.php/s/Mjk2hMjrJAyc5rX)
Put them in a checkpoints directory at the root of the project.

1. Testing both methods on the test dataset using the following command:

```bash
python testing_both_methods.py
```




## References

[1]: N. P. K. Cotter, T. W. Preist, and J. R. Sambles (1995). "Scattering-matrix approach to multilayer diffraction." Journal of the Optical Society of America A, 1097–1103. [Link to publication](https://opg.optica.org/viewmedia.cfm?r=1&rwjcode=josaa&uri=josaa-12-5-1097&html=true)

[2]: A. Rouxel, P. Gadras, "S-algorithm matrix repository" (2023). [Link to repository](https://gitlab.laas.fr/arouxel/s-algorithm) 

[3]: Calvez, S., Lafleur, G., et al. (2018). "Modelling anisotropic lateral oxidation from circular mesas." Optical Material Express. [Link to publication](https://opg.optica.org/ome/fulltext.cfm?uri=ome-8-7-1762&id=390232)

[4]: Monvoisin, N. et al. (2023). "Spectrally-shaped illumination for improved optical monitoring of lateral III-V-semiconductor oxidation." [Link to publication](https://opg.optica.org/oe/fulltext.cfm?uri=oe-31-8-12955&id=528801)
