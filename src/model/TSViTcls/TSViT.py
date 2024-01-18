# Copyright (c) 2023 Michail Tarasiou
# 
# Source: https://github.com/michaeltrs/DeepSatModels/blob/main/models/TSViT/TSViTcls.py
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
from einops.layers.torch import Rearrange
import numpy as np

from model.TSViTcls.module import Attention, PreNorm, FeedForward


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class TSViTcls(nn.Module):
    """
    Temporal-Spatial ViT for object classification (used in main results, section 4.3)
    """
    def __init__(self, model_config):
        super().__init__()
        self.image_size = model_config['img_res']
        self.patch_size = model_config['patch_size']
        self.num_patches_1d = self.image_size//self.patch_size
        self.num_classes = model_config['num_classes']
        self.num_frames = model_config['max_seq_len']
        self.dim = model_config['dim']
        if 'temporal_depth' in model_config:
            self.temporal_depth = model_config['temporal_depth']
        else:
            self.temporal_depth = model_config['depth']
        if 'spatial_depth' in model_config:
            self.spatial_depth = model_config['spatial_depth']
        else:
            self.spatial_depth = model_config['depth']
        # self.depth = model_config['depth']
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        self.dropout = model_config['dropout']
        self.emb_dropout = model_config['emb_dropout']
        self.pool = model_config['pool']
        self.scale_dim = model_config['scale_dim']
        assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_patches = self.num_patches_1d ** 2
        patch_dim = self.patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t (h p1) (w p2) -> (b h w) t (p1 p2)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(patch_dim, self.dim),)
        self.to_temporal_embedding_input = nn.Linear(self.num_frames, self.dim)
        self.temporal_token = nn.Parameter(torch.randn(1, self.num_classes, self.dim))
        self.temporal_transformer = Transformer(self.dim, self.temporal_depth, self.heads, self.dim_head,
                                                self.dim * self.scale_dim, self.dropout)
        self.space_token = nn.Parameter(torch.randn(1, self.num_classes, self.dim))
        self.space_pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dim))
        # print('space pos embedding: ', self.space_pos_embedding.shape)
        self.space_token = nn.Parameter(torch.randn(1, 1, self.dim))
        # print('space token: ', self.space_token.shape)
        self.space_transformer = Transformer(self.dim, self.spatial_depth, self.heads, self.dim_head, self.dim * self.scale_dim, self.dropout)
        self.dropout = nn.Dropout(self.emb_dropout)
        self.mlp_head = nn.Sequential(
            nn.Linear(self.dim*self.num_classes+6*self.num_frames, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes)
        )

    def forward(self, depth_images, imu_data):
        x = depth_images
        B, T, H, W = x.shape
        xt = x[:, :, 0, 0]
        # in the original paper, temporal embedding is the day of the year
        # here we use the frame number and don't slice away one channel
        # more info: https://github.com/michaeltrs/DeepSatModels/issues/4
        xt = (xt * T+0.0001).to(torch.int64)
        xt = F.one_hot(xt, num_classes=T).to(torch.float32)
        xt = xt.reshape(-1, T)
        temporal_pos_embedding = self.to_temporal_embedding_input(xt).reshape(B, T, self.dim)
        x = self.to_patch_embedding(x)
        x = x.reshape(B, -1, T, self.dim)
        x += temporal_pos_embedding.unsqueeze(1)
        x = x.reshape(-1, T, self.dim)
        cls_temporal_tokens = repeat(self.temporal_token, '() N d -> b N d', b=B * self.num_patches_1d ** 2)
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        x = self.temporal_transformer(x)
        x = x[:, :self.num_classes]
        x = x.reshape(B, self.num_patches_1d**2, self.num_classes, self.dim).permute(0, 2, 1, 3).reshape(B*self.num_classes, self.num_patches_1d**2, self.dim)
        x += self.space_pos_embedding#[:, :, :(n + 1)]
        x = self.dropout(x)
        cls_space_tokens = repeat(self.space_token, '() N d -> b N d', b=B * self.num_classes)
        x = torch.cat((cls_space_tokens, x), dim=1)
        x = self.space_transformer(x)
        x = x[:, 0]
        x = x.reshape(B, self.dim*self.num_classes)
        # concatenate x with imu data
        imu_data = imu_data.reshape(B, 6*self.num_frames)
        x = torch.cat((x, imu_data), dim=1)
        x = self.mlp_head(x)
        return x


if __name__ == "__main__":
    res = 24
    model_config = {'img_res': res, 'patch_size': 3, 'patch_size_time': 1, 'patch_time': 4, 'num_classes': 6,
                    'max_seq_len': 4, 'dim': 128, 'temporal_depth': 10, 'spatial_depth': 4, 'depth': 4,
                    'heads': 3, 'pool': 'cls', 'num_channels': 1, 'dim_head': 64, 'dropout': 0., 'emb_dropout': 0.,
                    'scale_dim': 4}

    model = TSViTcls(model_config)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)
    depth_imgs = torch.rand((8, 4, res, res))
    imu_data = torch.rand((8, 4, 6))
    out = model(depth_imgs, imu_data)
    print("Shape of out :", out.shape)  # [B, num_classes]
