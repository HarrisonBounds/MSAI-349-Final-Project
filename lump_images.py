import os
import shutil

# Path to the top-level directory
top_directory = "data/"

# Iterate through all subdirectories and files
for root, dirs, files in os.walk(top_directory):
    for file in files:
        # Get the full path of the file
        file_path = os.path.join(root, file)
        # Skip if the file is already in the top directory
        if root == top_directory:
            continue
        try:
            # Move the file to the top directory
            shutil.move(file_path, os.path.join(top_directory, file))
        except Exception as e:
            print(f"Could not move {file}: {e}")

# Optionally, remove empty directories
for root, dirs, _ in os.walk(top_directory, topdown=False):
    for dir_name in dirs:
        dir_path = os.path.join(root, dir_name)
        if not os.listdir(dir_path):  # Check if the directory is empty
            os.rmdir(dir_path)  # Remove the empty directory
