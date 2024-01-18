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
    res = 24
    model_config = {'img_res': res, 'patch_size': 3, 'patch_size_time': 1, 'patch_time': 2, 'num_classes': 6,
                    'max_seq_len': 4, 'dim': 128, 'temporal_depth': 10, 'spatial_depth': 4, 'depth': 4,
                    'heads': 3, 'pool': 'cls', 'num_channels': 1, 'dim_head': 64, 'dropout': 0., 'emb_dropout': 0.,
                    'scale_dim': 4}

    model = TSViTcls(model_config)
    depth_img = torch.rand((8, 4, res, res))
    imu_data = torch.rand((8, 4, 6))
    out = model(depth_img, imu_data)
    assert out.shape == (8, 6)
