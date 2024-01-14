import torch

from model.CNN import CNN

def test_cnn_forward_pass():
    model = CNN.CNN()
    input_data = torch.randn(32, 1, 64, 64)  # Batch size of 32, 1 channel, 64x64 images
    output = model(input_data)
    assert output.shape == torch.Size([32, 6])
