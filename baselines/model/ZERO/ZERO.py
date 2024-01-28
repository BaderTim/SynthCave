import torch
import torch.nn as nn
import numpy as np

class ZERO(nn.Module):
    def __init__(self, K=2):
        super(ZERO, self).__init__()

        self.dataset_type = "image"
        self.K = K

    def forward(self, depth_images, imu_data=None):
        return torch.tensor([0, 0, 0, 0, 0]).repeat(depth_images.shape[0], 1).float().to(depth_images.device)


if __name__ == "__main__":

    K = 1

    model = ZERO(K=K)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    # Example input tensor (replace with your actual data)
    depth_images = torch.randn(16, K, 1, 80, 240)  

    # Forward pass
    output = model(depth_images)
    print("Shape of out :", output.shape)
