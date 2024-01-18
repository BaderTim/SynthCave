import torch

from model.CNN import CNN

def test_cnn_forward_pass():
    input_seq_len = 2
    model = CNN.CNN(input_seq_len=input_seq_len)
    depth_images = torch.randn(4, 1, 64, 64*input_seq_len) 
    imu_data = torch.randn(4, input_seq_len, 6) 
    output = model(depth_images, imu_data)
    assert output.shape == torch.Size([4, 6])
