# Copyright (c) Hehe Fan, Xin Yu, Yuhang Ding, Yi Yang, Mohan Kankanhalli
# 
# Source: https://github.com/hehefan/Point-Spatio-Temporal-Convolution/blob/main/models/sequence_classification.py
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.PSTNet.modules.pst_convolutions import PSTConv


class NTU(nn.Module):
    def __init__(self, radius=0.1, nsamples=3*3, K=4):
        super(NTU, self).__init__()

        self.K = K
        self.dataset_type = "point"

        self.temporal_padding_for_b = [1, 2] if self.K > 2 else [1, 1]

        self.conv1 =  PSTConv(in_planes=0,
                              mid_planes=45,
                              out_planes=64,
                              spatial_kernel_size=[radius, nsamples],
                              temporal_kernel_size=1,
                              spatial_stride=2,
                              temporal_stride=1,
                              temporal_padding=[0,0])

        self.conv2a = PSTConv(in_planes=64,
                              mid_planes=96,
                              out_planes=128,
                              spatial_kernel_size=[2*radius, nsamples],
                              temporal_kernel_size=3,
                              spatial_stride=2,
                              temporal_stride=2,
                              temporal_padding=[1,2])

        self.conv2b = PSTConv(in_planes=128,
                              mid_planes=192,
                              out_planes=256,
                              spatial_kernel_size=[2*radius, nsamples],
                              temporal_kernel_size=3,
                              spatial_stride=1,
                              temporal_stride=1,
                              temporal_padding=self.temporal_padding_for_b)

        self.conv3a = PSTConv(in_planes=256,
                              mid_planes=384,
                              out_planes=512,
                              spatial_kernel_size=[2*2*radius, nsamples],
                              temporal_kernel_size=3,
                              spatial_stride=2,
                              temporal_stride=2,
                              temporal_padding=[1,2])

        self.conv3b = PSTConv(in_planes=512,
                              mid_planes=768,
                              out_planes=1024,
                              spatial_kernel_size=[2*2*radius, nsamples],
                              temporal_kernel_size=3,
                              spatial_stride=1,
                              temporal_stride=1,
                              temporal_padding=self.temporal_padding_for_b)

        self.conv4 =  PSTConv(in_planes=1024,
                              mid_planes=1536,
                              out_planes=2048,
                              spatial_kernel_size=[2*2*radius, nsamples],
                              temporal_kernel_size=1,
                              spatial_stride=2,
                              temporal_stride=1,
                              temporal_padding=[0,0])

        # MLP head
        self.mlp_head = nn.Sequential(
            nn.Linear(2048 + 6*K, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )


    def forward(self, xyzs, imu_data):

        B, K, C, N = xyzs.shape

        new_xys, new_features = self.conv1(xyzs, None)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv2a(new_xys, new_features)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv2b(new_xys, new_features)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv3a(new_xys, new_features)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv3b(new_xys, new_features)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv4(new_xys, new_features)               # (B, L, C, N)

        new_features = torch.mean(input=new_features, dim=-1, keepdim=False)    # (B, L, C)

        new_feature = torch.max(input=new_features, dim=1, keepdim=False)[0]    # (B, C)

        imu_data = imu_data.reshape(B, 6*self.K)
        x = torch.cat((new_feature, imu_data), dim=1)

        out = self.mlp_head(x)

        return out

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    K = 16
    model = NTU(K=K).to(device)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    x = torch.randn(4, K, 128, 3).to(device) # (B, L, C, N)
    imu_data = torch.randn(4, K, 6).to(device) # (B, 6*K)

    out = model(x, imu_data)

    print("Shape of out :", out.shape) 
