# Based on the best performing implementation from Austin Nicolai, Ryan Skeele, Christopher Eriksen, and Geoffrey A. Hollinger
# Paper: https://research.engr.oregonstate.edu/rdml/sites/research.engr.oregonstate.edu.rdml/files/final_deep_learning_lidar_odometry.pdf

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class CNN(nn.Module):
    def __init__(self, K=2):
        super(CNN, self).__init__()

        self.dataset_type = "image"

        self.K = K
        self.height = 80
        self.width = 240*self.K

        # Feature extractor layers (R = Kernel size)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=int(2+np.log(self.K)), stride=int(2+np.log(self.K)), padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(int(self.height/(2*int(2+np.log(self.K)))) * int(self.width/(2*int(2+np.log(self.K)))) * 64 + 6*self.K, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 5)  # Assuming regression for odometry estimation

    def forward(self, depth_images, imu_data):
        B, K, C, H, W = depth_images.shape
        # -> B, C, H, W*K
        depth_image = depth_images.reshape(B, C, H, W*K)
        # Feature extractor layers
        x = F.relu(self.conv1(depth_image))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        # Flatten before fully connected layers
        x = x.reshape(B, -1)

        # Concatenate IMU data
        # concatenate x with imu data
        imu_data = imu_data.reshape(B, 6*self.K)
        x = torch.cat((x, imu_data), dim=1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x


if __name__ == "__main__":

    K = 16

    model = CNN(K=K)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    # Example input tensor (replace with your actual data)
    depth_images = torch.randn(16, K, 1, 80, 240)  
    imu_data = torch.randn(16, K, 6) 

    # Forward pass
    output = model(depth_images, imu_data)
    print("Shape of out :", output.shape)
