import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import ASTGCN


class ASTGCN_Model(nn.Module):
    def __init__(self, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, num_for_predict, len_input, num_of_vertices, normalization=None, bias=True, mlp_hidden_dim=64, mlp_output_dim=1):
        super(ASTGCN_Model, self).__init__()

        # Feature extractor using ASTGCN layers
        self.astgcn = ASTGCN(
            nb_block=nb_block,
            in_channels=in_channels,
            K=K,
            nb_chev_filter=nb_chev_filter,
            nb_time_filter=nb_time_filter,
            time_strides=time_strides,
            num_for_predict=num_for_predict,
            len_input=len_input,
            num_of_vertices=num_of_vertices,
            normalization=normalization,
            bias=bias
        )

        # MLP block
        self.mlp = nn.Sequential(
            nn.Linear(num_of_vertices * mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_output_dim)
        )

    def forward(self, X, edge_index):
        # Reshape X to match the ASTGCN input format
        X = X.permute(0, 2, 3, 1).contiguous()  # (B, F_in, T_in, N_nodes)
        X = X.view(X.size(0), X.size(1), -1)  # (B, F_in, T_in * N_nodes)

        # Extract features using ASTGCN layers
        astgcn_output = self.astgcn(X, edge_index)

        # Flatten the output for the MLP block
        flattened_output = astgcn_output.view(astgcn_output.size(0), -1)

        # Pass through the MLP block
        mlp_output = self.mlp(flattened_output)

        return mlp_output


if __name__ == "__main__":
    # Initialize model with desired parameters
    model = ASTGCN_Model(nb_block=3, in_channels=64, K=3, nb_chev_filter=64, nb_time_filter=64, time_strides=2, num_for_predict=12, len_input=12, num_of_vertices=10, normalization='sym', bias=True)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    # Example input tensor and edge index
    X = torch.randn(32, 10, 64, 12)  # Batch size of 32, 10 vertices, 64 channels, 12 time periods
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                           [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]], dtype=torch.long)

    # Forward pass
    output = model(X, edge_index)
    print("Shape of out :", output.shape)
