import torch
import math
from PIL import Image

# Load the original image
# input_file = "path/to/your/image.png"  # Replace with your image path
input_file = 'user_sketch.png'
original_image = Image.open(input_file)

# List of rotation angles
angles = [90, 180, 270]

# Save the rotated images
for angle in angles:
    rotated_image = original_image.rotate(angle, expand=True)
    output_file = f"modified_image_{angle}.png"  # Replace with your save path
    rotated_image.save(output_file)
    print(f"Saved rotated image: {output_file}")
