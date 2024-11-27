import torch
from torchvision import transforms
from dataset import CustomImageDataset
from torch.utils.data import DataLoader
from cnn import CNN
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
# from sketch_interface import Interface


# Interface = Interface()
# Interface.draw()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE} device")

# Loss lists for plotting
train_losses = []
valid_losses = []
test_losses = []

# Hyperparameters
batch_size = 32
image_width = 300
image_height = 300
lr = 0.001
epochs = 2

# Loading and pre-processing data
transform = transforms.Compose([
    transforms.Resize((image_width, image_height)),
])

dataset = CustomImageDataset("sketches.csv", "data", transform)
train_set, valid_set = torch.utils.data.random_split(dataset, [0.75, 0.25])
test_image = "user_sketch.png"

train_loader = DataLoader(train_set, batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size, shuffle=True)

model = CNN(image_width, image_height).to(DEVICE)
# Tune this for potential better results
optimizer = torch.optim.Adam(model.parameters(), lr)
loss_func = nn.CrossEntropyLoss()

# Training Loop
for i in tqdm(range(epochs)):
    # print(f"Epoch {i}\n=========================")
    model.train()
    num_samples_processed = 0
    epoch_loss = 0
    for b, (im, target) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
        image, y_true = im.to(DEVICE), target.to(DEVICE)

        y_pred = model(image)
        batch_loss = loss_func(y_pred, y_true)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        num_samples_processed += y_true.shape[0]
        # print(f"Batch Loss: {batch_loss.item()} [{num_samples_processed}/{len(train_loader.dataset)}]")
        epoch_loss = (epoch_loss * b + batch_loss.item()) / (b + 1)
        torch.save(model.state_dict(), f"model_epoch_{i}.pth")

# Save the final model.
torch.save(model.state_dict(), "final_model.pth")
print("Final model saved to 'final_model.pth'")

# Validation loop
# Visualisation section

# Testing loop

test_transform = transforms.Compose([
    # Resize to model's input size
    transforms.Resize((image_width, image_height)),
    transforms.ToTensor(),  # Convert PIL to tensor
])

# Load the trained model
model.load_state_dict(torch.load("final_model.pth"))
model.eval()  # Set the model to evaluation mode
print("testing loop...")
# Process the image and make a prediction
with torch.no_grad():  # Disable gradient computation for inference
    # Load and preprocess the image
    image = Image.open(test_image).convert("L")
    image = test_transform(image)
    # Add one axis to the batch dimension
    image = image.unsqueeze(0).to(DEVICE)

    # Get the model's prediction
    output = model(image)
    # Get the class index with the highest score
    predicted_label = torch.argmax(output, dim=1).item()
    print(f"Predicted Label for '{test_image}': {predicted_label} : ")

    for key, value in dataset.label_map.items():
        if value == predicted_label:
            key_for_value = key
            print(f"the image is of : {key}")
            break
