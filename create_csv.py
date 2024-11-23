import os
import pandas as pd

# Define the root directory containing the class subdirectories
root_dir = 'data/'

# Initialize a list to hold the data
data = []

# Loop through each class directory
for class_name in os.listdir(root_dir):
    class_dir = os.path.join(root_dir, class_name)
    if os.path.isdir(class_dir):  # Ensure it's a directory
        # Loop through each file in the class directory
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            if os.path.isfile(file_path):  # Ensure it's a file
                # Remove the root directory prefix explicitly
                relative_path = file_path[len(root_dir):]
                # Append relative path and class name as label
                data.append([relative_path, class_name])

# Create a DataFrame
df = pd.DataFrame(data, columns=['file_path', 'label'])

# Save to a CSV file
df.to_csv('sketches.csv', index=False)
