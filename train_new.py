import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from cnn import CNN
import torch.nn as nn
import numpy as np
from datetime import datetime
from common import *
from utils import config
import utils
# from sketch_interface import Interface
from sklearn.metrics import accuracy_score, precision_score
import matplotlib.pyplot as plt

# Interface = Interface()
# Interface.draw()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE} device")

# Loss lists for plotting
train_losses = []
valid_losses = []
test_losses = []

resize_width = 120
resize_height = 120
# Hyperparameters
batch_size = 64
lr = 0.001
epochs = 15

# Loading and pre-processing data
transform = transforms.Compose([
    transforms.Resize((resize_width, resize_height)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5],)
])

dataset = datasets.ImageFolder("data", transform=transform)
train_set, valid_set, test_set = torch.utils.data.random_split(dataset, [
                                                               0.75, 0.15, 0.10])
test_image = "user_sketch.png"

train_loader = DataLoader(train_set, batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size, shuffle=True)

model = CNN(resize_width, resize_height)
# Tune this for potential better results
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

print("Number of float-valued parameters:", count_parameters(model))

# Attempts to restore the latest checkpoint if exists
print("Loading source...")
model, start_epoch, stats = restore_checkpoint(
    model, config("cnn.checkpoint"))

axes = utils.make_training_plot("CNN Training")

evaluate_epoch(
    axes,
    train_loader,
    valid_loader,
    test_loader,
    model,
    loss_func,
    start_epoch,
    stats,
    multiclass=True
)

# initial val loss for early stopping
global_min_loss = stats[0][1]

# TODO: patience for early stopping
patience = 10
curr_count_to_patience = 0
#

# Loop over the entire dataset multiple times
epoch = start_epoch
while curr_count_to_patience < patience:
    # Train model
    train_epoch(train_loader, model, loss_func, optimizer)

    # Evaluate model
    evaluate_epoch(
        axes,
        train_loader,
        valid_loader,
        test_loader,
        model,
        loss_func,
        epoch + 1,
        stats,
        multiclass=True,
    )

    # Save model parameters
    save_checkpoint(model, epoch + 1, config("cnn.checkpoint"), stats)

    curr_count_to_patience, global_min_loss = early_stopping(
        stats, curr_count_to_patience, global_min_loss
    )
    epoch += 1

# Save figure and keep plot open
print("Finished Training")
utils.save_source_training_plot()
utils.hold_training_plot()
now = datetime.now()
formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
torch.save(model.state_dict(), f'models/{formatted_time}_sketch_cnn.pth')
print("Training complete and model saved.")
