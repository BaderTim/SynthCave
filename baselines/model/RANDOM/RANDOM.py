import torch
import torch.nn as nn
import numpy as np

class RANDOM(nn.Module):
    def __init__(self, K=2):
        super(RANDOM, self).__init__()

        self.dataset_type = "image"
        self.K = K

    def forward(self, depth_images, imu_data=None):
        # return 5 random values between -1 and 1
        x = np.random.uniform(-1, 1, size=(depth_images.shape[0], 5))
        return torch.tensor(x).float().to(depth_images.device)


if __name__ == "__main__":

    K = 1

    model = RANDOM(K=K)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    # Example input tensor (replace with your actual data)
    depth_images = torch.randn(16, K, 1, 80, 240)  

    # Forward pass
    output = model(depth_images)
    print("Shape of out :", output.shape)
