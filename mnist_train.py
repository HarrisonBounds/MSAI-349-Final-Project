import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn
from cnn import CNN

# Hyperparameters
batch_size = 16
epochs = 5
learning_rate = 0.001

# Define transformation (convert images to tensor and normalize)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure the images are grayscale
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize images to the range [-1, 1]
])

# Load MNIST dataset
train_set = datasets.ImageFolder(
    root='mnist_data/trainingSet/trainingSet',
    transform=transform
)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)


# Initialize model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Is cuda available?: ", torch.cuda.is_available())
print("Number of available GPUs : ", torch.cuda.device_count())
model = CNN(28, 28).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

# Save the trained model
torch.save(model.state_dict(), 'mnist_cnn.pth')

print("Training complete and model saved.")
