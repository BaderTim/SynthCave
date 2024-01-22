import torch

from model.CNN import CNN

def test_cnn_forward_pass():
    K = 2
    model = CNN.CNN(K=K)
    depth_images = torch.randn(4, K, 1, 80, 240) 
    imu_data = torch.randn(4, K, 6) 
    output = model(depth_images, imu_data)
    assert output.shape == torch.Size([4, 5])
