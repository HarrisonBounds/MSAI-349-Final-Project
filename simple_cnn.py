import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class CNN(nn.Module):
    def __init__(self, width, height):
        super().__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(
            3, 3), stride=(2, 2), padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(
            3, 3), stride=(2, 2), padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(
            3, 3), stride=(2, 2), padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.fc1 = nn.Linear(64 * 4 * 4, 250)
        self.fc2 = nn.Linear(250, 250)

        self.dropout = nn.Dropout(0.5)

        self.init_weights()

    def init_weights(self):
        torch.manual_seed(42)
        for conv in [self.conv1, self.conv2, self.conv3]:
            C_in = conv.weight.size(1)
            nn.init.kaiming_normal_(
                conv.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(conv.bias, 0.0)

        nn.init.normal_(self.fc1.weight, 0.0, 2.0 / (64*4*4 + 250))
        nn.init.constant_(self.fc1.bias, 0.0)

    def forward(self, x):
        conv1_out = self.conv1(x)
        bn1_out = self.bn1(conv1_out)
        relu1_out = F.relu(bn1_out)
        pool1_out = self.pool(relu1_out)

        conv2_out = self.conv2(pool1_out)
        bn2_out = self.bn2(conv2_out)
        relu2_out = F.relu(bn2_out)
        pool2_out = self.pool(relu2_out)

        conv3_out = self.conv3(pool2_out)
        bn3_out = self.bn3(conv3_out)
        relu3_out = F.relu(bn3_out)

        pooled = self.adaptive_pool(relu3_out)

        flattened = torch.flatten(pooled, 1)
        fc1_out = self.fc1(flattened)
        fc1_out = self.dropout(fc1_out)
        z = self.fc2(fc1_out)

        return z
