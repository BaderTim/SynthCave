import torch

from model.RANDOM.RANDOM import RANDOM

def test_random_forward_pass():
    K = 2
    model = RANDOM(K=K)
    depth_images = torch.randn(4, K, 1, 80, 240) 
    imu_data = torch.randn(4, K, 6) 
    out = model(depth_images, imu_data)
    assert out.shape == (4, 5)
