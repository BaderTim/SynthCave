import torch

from baselines.model.ZERO.ZERO import ZERO

def test_zero_forward_pass():
    K = 2
    model = ZERO(K=K)
    depth_images = torch.randn(4, K, 1, 80, 240) 
    imu_data = torch.randn(4, K, 6) 
    out = model(depth_images, imu_data)
    assert out.shape == (8, 5)
    assert out == torch.zeros((K, 5))
