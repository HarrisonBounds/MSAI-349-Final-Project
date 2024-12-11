import torch
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import pandas as pd
import numpy as np

# Define the transform (resize, convert to tensor, normalize)
transform = transforms.Compose([
    # resize the image to 32x32 (or any size you need)
    transforms.Resize((32, 32)),
    transforms.ToTensor(),  # convert image to a tensor
])

# Load the dataset
dataset = datasets.ImageFolder("data", transform=transform)

# Prepare the data for conversion
image_data = []
labels = []

# Iterate through the dataset
for img, label in dataset:
    # Flatten the image (convert it to a 1D array of pixel values)
    flattened_image = img.view(-1).numpy()  # Flatten the image tensor
    image_data.append(flattened_image)
    labels.append(label)

# Convert lists to numpy arrays
image_data = np.array(image_data)
labels = np.array(labels)

# Create a DataFrame
df = pd.DataFrame(image_data)
df.insert(0, 'label', labels)  # Add labels as the last column

# Save to CSV
df.to_csv('data/image_data.csv', index=False)
