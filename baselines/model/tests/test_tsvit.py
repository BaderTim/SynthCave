import torch
import torch.nn as nn
from model.TSViTcls.module import FeedForward, Attention, PreNormLocal
from model.TSViTcls.TSViT import TSViTcls

def test_feedforward_forward():
    dim, hidden_dim = 32, 64
    model = FeedForward(dim, hidden_dim)
    x = torch.randn((16, 32))
    output = model.forward(x)
    assert output.shape == (16, 32)

def test_attention_forward():
    dim, heads, dim_head = 32, 8, 64
    model = Attention(dim, heads, dim_head)
    x = torch.randn((16, 64, 32))
    output = model.forward(x)
    assert output.shape == (16, 64, 32)

def test_prenorm_local():
    dim = 128
    fn = nn.Linear(dim, dim)
    model = PreNormLocal(dim, fn)
    x = torch.rand(1, dim, 1, dim)  
    output = model(x)
    assert output.shape == x.shape

def test_tsvitcls_forward_call():
    model = TSViTcls(K=4)
    depth_img = torch.rand((8, 4, 1, 80, 240))
    imu_data = torch.rand((8, 4, 6))
    out = model(depth_img, imu_data)
    assert out.shape == (8, 6)
