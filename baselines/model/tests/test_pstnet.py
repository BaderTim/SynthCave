import torch

from model.PSTNet.PSTNet import NTU

def test_pstnet_forward_call():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device == torch.device("cuda"), "CUDA not available, PSTNet does not support CPU"
    model = NTU(K=4).to(device)
    x = torch.randn(4, 4, 128, 3).to(device) # (B, L, C, N)
    out = model(x).to("cpu")
    assert out.shape == (4, 6)
