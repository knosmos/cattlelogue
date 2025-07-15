# simple CNN for livestock projections

import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_VECTOR_SIZE = 52


class CNN(nn.Module):
    def __init__(self, in_channels=INPUT_VECTOR_SIZE, out_channels=1):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(2048, 256)  # Assuming input size is 32x32
        self.fc2 = nn.Linear(256, out_channels)  # Binary classification

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
