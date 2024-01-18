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
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'src\model\PSTNet\modules'))

from modules.pst_convolutions import PSTConv


class NTU(nn.Module):
    def __init__(self, radius=0.1, nsamples=3*3, num_classes=20):
        super(NTU, self).__init__()

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
                              temporal_padding=[0,0])

        self.conv2b = PSTConv(in_planes=128,
                              mid_planes=192,
                              out_planes=256,
                              spatial_kernel_size=[2*radius, nsamples],
                              temporal_kernel_size=3,
                              spatial_stride=1,
                              temporal_stride=1,
                              temporal_padding=[0,0])

        self.conv3a = PSTConv(in_planes=256,
                              mid_planes=384,
                              out_planes=512,
                              spatial_kernel_size=[2*2*radius, nsamples],
                              temporal_kernel_size=3,
                              spatial_stride=2,
                              temporal_stride=2,
                              temporal_padding=[0,0])

        self.conv3b = PSTConv(in_planes=512,
                              mid_planes=768,
                              out_planes=1024,
                              spatial_kernel_size=[2*2*radius, nsamples],
                              temporal_kernel_size=3,
                              spatial_stride=1,
                              temporal_stride=1,
                              temporal_padding=[0,0])

        self.conv4 =  PSTConv(in_planes=1024,
                              mid_planes=1536,
                              out_planes=2048,
                              spatial_kernel_size=[2*2*radius, nsamples],
                              temporal_kernel_size=1,
                              spatial_stride=2,
                              temporal_stride=1,
                              temporal_padding=[0,0])

        self.fc = nn.Linear(2048, num_classes)

    def forward(self, xyzs):

        print("initial shape of xyzs :", xyzs.shape)
        new_xys, new_features = self.conv1(xyzs, None)
        new_features = F.relu(new_features)
        print("shape of new_features after conv1:", new_features.shape)
        print("shape of new_xys after conv1:", new_xys.shape)

        new_xys, new_features = self.conv2a(new_xys, new_features)
        new_features = F.relu(new_features)
        print("shape of new_features after conv2a:", new_features.shape)
        print("shape of new_xys after conv2a:", new_xys.shape)

        new_xys, new_features = self.conv2b(new_xys, new_features)
        new_features = F.relu(new_features)
        print("shape of new_features after conv2b:", new_features.shape)
        print("shape of new_xys after conv2b:", new_xys.shape)

        new_xys, new_features = self.conv3a(new_xys, new_features)
        new_features = F.relu(new_features)
        print("shape of new_features after conv3a:", new_features.shape)
        print("shape of new_xys after conv3a:", new_xys.shape)

        new_xys, new_features = self.conv3b(new_xys, new_features)
        new_features = F.relu(new_features)
        print("shape of new_features after conv3b:", new_features.shape)
        print("shape of new_xys after conv3b:", new_xys.shape)

        new_xys, new_features = self.conv4(new_xys, new_features)               # (B, L, C, N)
        print("shape of new_features after conv4:", new_features.shape)

        new_features = torch.mean(input=new_features, dim=-1, keepdim=False)    # (B, L, C)

        new_feature = torch.max(input=new_features, dim=1, keepdim=False)[0]    # (B, C)

        out = self.fc(new_feature)

        return out

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = NTU().to(device)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    x = torch.randn(4, 5, 2048, 3).to(device) # (B, L, C, N)

    out = model(x)

    print("Shape of out :", out.shape) 