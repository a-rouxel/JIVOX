"""
VCSEL Configuration Generator Script

Overview:
This script generates a series of configuration files for VCSEL (Vertical-Cavity Surface-Emitting Laser) simulations.
The configurations vary in VCSEL type, dimensions, location, rotation, vertices count, and bevel segments. Each
configuration is saved as a JSON file in a specified directory.


Features:
- Generates a predefined number of VCSEL configurations.
- Supports two VCSEL types: 'cube' and 'cylinder' (for the cylinder, it can be a hexagone or a disk depending on the vertices number).
- Randomizes scale, location, and rotation for diverse configurations.
- Saves each configuration as a JSON file in a structured directory.

Usage:
1. Set Parameters: Modify the script's initial parameters to suit your needs.
2. Run the Script: Execute the script in a Python environment to generate the configurations.

Script Workflow:
1. Configuration Generation: Randomly selects VCSEL type and randomizes various parameters.
2. Directory Management: Checks and creates directories for each configuration.
3. File Writing: Saves the configurations as JSON files in their respective directories.

Output:
- Directories for each configuration containing a 'config.json' file with the VCSEL parameters.

Example JSON Structure:
{
    "name": "VCSEL_0",
    "type": "cube",
    "scale_pad": [1.7, 1.7, 0.1],
    "scale_aperture": [1.6, 1.6, 0.1],
    "scale_mesa": [1.6, 1.6, 0.1],
    "location": [0.1, -0.05, 0],
    "rotation": [0, 0, 45],
    "vertices": 20,
    "bevel_segments": 3
}

"""
import random
import json
import os
import yaml

# Load the configuration from 'config.yaml'
with open('dataset_generation/config_dataset.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Assigning parameters from the configuration file
starting_idx = config['starting_idx']
number_of_configs_to_generate = config['number_of_configs_to_generate']
dir_configs = config['dir_dataset']
vcsel_types = config['vcsel_types']
basic_scale_mesa = tuple(config['basic_scale_mesa'])
rotation_ = config['rotation_']
location = config['location']
vertices = config['vertices']
bevel_segments = config['bevel_segments']

for idx in range(starting_idx, number_of_configs_to_generate + starting_idx):
    vcsel_type = random.choice(vcsel_types)
    add_scale_mesa = random.uniform(-0.5, 0)
    scale_mesa = [basic_scale_mesa[0] + add_scale_mesa, basic_scale_mesa[1] + add_scale_mesa, basic_scale_mesa[2]]

    if rotation_ == "random":
        rotation = [0, 0, random.uniform(0, 360)]
    elif rotation_ == "None":
        rotation = [0, 0, 0]

    location = [random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2), 0]
    vertices_ = random.choice(vertices)
    bevel_segments_ = random.choice(bevel_segments)
    add_scale = random.uniform(0, 0.04)
    scale_pad = [scale_mesa[0] + add_scale, scale_mesa[1] + add_scale, scale_mesa[2]]

    # VCSEL parameters
    vcsel_params = {
        "name": f"VCSEL_{idx}",
        "type": vcsel_type,
        "scale_pad": scale_pad,
        "scale_aperture": scale_mesa,
        "scale_mesa": scale_mesa,
        "location": location,
        "rotation": rotation,
        "vertices": vertices_,
        "bevel_segments": bevel_segments_
    }

    # Directory for saving renders
    output_directory = os.path.join(dir_configs, vcsel_params["name"])
    os.makedirs(output_directory, exist_ok=True)

    # Save the configuration to a JSON file
    config_file_path = os.path.join(output_directory, 'config.json')
    with open(config_file_path, 'w') as config_file:
        json.dump(vcsel_params, config_file, indent=4)
