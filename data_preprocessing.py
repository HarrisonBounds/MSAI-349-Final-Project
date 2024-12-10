import torch
import math
from PIL import Image
import os

# Take data file as input
# input_file = "path/to/your/image.png"  # Replace with your image path
input_folder_name = "modTest"  #mnist/trainingSet

# Create the modified folder.
modified_folder_name = f"modified_{input_folder_name}"
try:
    os.makedirs(modified_folder_name, exist_ok=True)  
    print(f"Folder created: {modified_folder_name}")
except OSError as e:
    print(f"Error creating folder: {e}")


# Loop through all subdirectories and files
for root, dirs, files in os.walk(input_folder_name):
    for file in files:
        

        if file.lower().endswith(('.png','.jpg')):  # Check for image files
            input_file_name = file
            input_sub_fold = root.replace(f"{input_folder_name}/","")
            input_file_path = os.path.join(root, file)

            # create sub-folder if not already present.
            try:
                sub_fold_path = os.path.join(modified_folder_name,input_sub_fold)
                os.makedirs(sub_fold_path, exist_ok=True)  
                print(f"Folder created: {sub_fold_path}")
            except OSError as e:
                print(f"Error creating folder: {e}")

            original_image = Image.open(input_file_path)

            # List of rotation angles
            angles = [0, 90, 180, 270]

            # Save the rotated images
            for angle in angles:
                rotated_image = original_image.rotate(angle, expand=True)
                output_file_path_root = os.path.join(modified_folder_name, input_sub_fold)
                print(f"root: {output_file_path_root}")
                output_file_name = f"modified_{input_file_name}_{angle}.png" 
                output_file_path = os.path.join(output_file_path_root , output_file_name)
                print(f"Saved rotated image: {output_file_path}")
                rotated_image.save(output_file_path)


