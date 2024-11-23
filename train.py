import torch
from torchvision import transforms
from dataset import CustomImageDataset
from torch.utils.data import DataLoader
from cnn import CNN
import torch.nn as nn


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE} device")

#Loss lists for plotting
train_losses = []
valid_losses = []
test_losses = []

#Hyperparameters
batch_size = 32
image_width = 300
image_height = 300
lr = 0.001
epochs = 20

transform = transforms.Compose([
    transforms.Resize((image_width, image_height)),
])

dataset = CustomImageDataset("sketches.csv", "data", transform)

train_set, valid_set, test_set = torch.utils.data.random_split(dataset, [0.7, 0.15, 0.15])

train_loader = DataLoader(train_set, batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size, shuffle=True)

model = CNN(image_width, image_height)
optimizer = torch.optim.Adam(model.parameters(), lr) #Tune this for potential better results
loss = nn.CrossEntropyLoss()

#Training Loop
for i in range(epochs):
    print(f"Epoch {i}\n=========================")
    model.train()
    




