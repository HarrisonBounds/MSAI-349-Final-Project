import torch
from torchvision import transforms
from dataset import CustomImageDataset
from torch.utils.data import DataLoader

#hyperparameters
batch_size = 32

transform = transforms.Compose([
    transforms.Resize((300, 300)),
])

dataset = CustomImageDataset("sketches.csv", "data", transform)

train_set, valid_set, test_set = torch.utils.data.random_split(dataset, [0.7, 0.15, 0.15])

train_loader = DataLoader(train_set, batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size, shuffle=True)

