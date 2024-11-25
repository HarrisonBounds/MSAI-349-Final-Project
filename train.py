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
lr = 0.01
epochs = 20

transform = transforms.Compose([
    transforms.Resize((image_width, image_height)),
])

dataset = CustomImageDataset("sketches.csv", "data", transform)

train_set, valid_set, test_set = torch.utils.data.random_split(dataset, [0.7, 0.15, 0.15])

train_loader = DataLoader(train_set, batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size, shuffle=True)

model = CNN(image_width, image_height).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr) #Tune this for potential better results
loss_func = nn.CrossEntropyLoss()

#Training Loop


for i in range(epochs):
    print(f"Epoch {i}\n=========================")
    model.train()
    num_samples_processed = 0
    epoch_loss = 0
    for b, (im, target) in enumerate(train_loader):
        image, y_true = im.to(DEVICE), target.to(DEVICE)
        
        y_pred = model(image)
        batch_loss = loss_func(y_pred, y_true)
        
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        
        num_samples_processed += y_true.shape[0]
        
        print(f"Batch Loss: {batch_loss.item()} [{num_samples_processed}/{len(train_loader.dataset)}]")
        
        epoch_loss = (epoch_loss * b + batch_loss.item()) / (b + 1)
        
        
        
        
        
        




