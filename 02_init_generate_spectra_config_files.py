import json
import os
import random
import yaml

# Function to generate a single layer entry with modified thickness
def modify_layer(layer_entry, new_thickness):
    modified_layer = layer_entry.copy()
    if modified_layer["type"] == "DBR":
        i = random.randint(1, 2)  # Randomly select the first or second layer
        modified_layer[f"layer{i}"]["thickness"] = round(new_thickness, 2)
    else:
        modified_layer["thickness"] = round(new_thickness, 2)  # Round to 2 decimal places
    return modified_layer

with open('dataset_generation/config_dataset.yaml', 'r') as file:
    config = yaml.safe_load(file)

input_dir_path = "dataset_generation/multilayer_stacks/configs/"
output_dir_path = config["dir_dataset"] + "/output_spectra/configs/"

#
# filename = "stacks_VCSEL940_non_resonnant"
# filename = "stacks_VCSEL850"
filename = "stacks_VCSEL940_resonnant"


for j in range(100):

    # Load the JSON data from the file
    with open(input_dir_path + filename + "/" + filename + ".json", "r") as json_file:
        data = json.load(json_file)


    # Specify multiple nth layers to modify (e.g., [1, 3] for the second and fourth layers)
    # nth_layers = [1,3,4,6] # VCSEL 940
    nth_layers = [3,5,7,9,11] # VCSEL 850


    # new_thickness = [random.uniform(0.056, 0.075),
    #                  random.uniform(0.025, 0.035),
    #                  random.uniform(0.065, 0.1),
    #                  random.uniform(0.065, 0.1),
    #                  random.uniform(0.056, 0.075),
    #                  ]  # Generate random thickness values at 850 nm

    new_thickness = [random.uniform(0.056, 0.075),
                     random.uniform(0.025, 0.035),
                     random.uniform(0.189, 0.21),
                     random.uniform(0.056, 0.075)
                     ]  # Generate random thickness values at 940 nm

    # Loop over each stack, skipping 'stack_background'
    for stack_name, stack_layers in data.items():
            if stack_name == 'stack_background':
                continue
            if stack_name == 'stack_limit' :
                continue

            for idx_layer,nth_layer in enumerate(nth_layers):
                if len(stack_layers) > nth_layer:  # Check if the stack has enough layers
                    # Generate a random thickness value within your desired range for this specific layer
                    new_thick = new_thickness[idx_layer]
                    modified_layer = modify_layer(stack_layers[nth_layer], new_thick)
                    data[stack_name][nth_layer] = modified_layer
                    print(f"Modified layer {nth_layer + 1} in {stack_name} with new thickness: {new_thick:.2f}")

    # Save the modified JSON to a file
    os.makedirs(output_dir_path + filename, exist_ok=True)
    with open(output_dir_path + filename + "/" + f"stacks_{j}.json", "w") as modified_file:
        json.dump(data, modified_file, indent=2)

    print("Modified JSON has been saved to 'modified_stacks.json'.")