import torch
import torch.nn as nn
import numpy as np
from torch_geometric_temporal.nn.attention.astgcn import ASTGCNBlock
from torch_geometric.nn import TopKPooling


class ASTGCN(nn.Module):
    def __init__(self, node_features=3, input_seq_length=4, node_count=100, blocks=3, pooling_ratio=1.0):
        super(ASTGCN, self).__init__()

        self.pooling_ratio = pooling_ratio
        self.node_features = node_features
        self.input_seq_length = input_seq_length
        self.node_count = int(node_count * pooling_ratio)
        self.blocks = blocks

        self.time_strides = 1

        # Graph pooling layer
        self.graph_pooling = TopKPooling(in_channels=self.node_features, ratio=self.pooling_ratio)

        # Feature extractor using ASTGCN layers
        self.feature_extractor = nn.ModuleList(
            [
                ASTGCNBlock(
                    in_channels=self.node_features,
                    K=3,
                    nb_chev_filter=64,
                    nb_time_filter=64,
                    time_strides=self.time_strides,
                    num_of_vertices=self.node_count,
                    num_of_timesteps=self.input_seq_length
                )
            ]
        )
        self.feature_extractor.extend(
            [
                ASTGCNBlock(
                    in_channels=64,
                    K=3,
                    nb_chev_filter=64,
                    nb_time_filter=64,
                    time_strides=self.time_strides,
                    num_of_vertices=self.node_count,
                    num_of_timesteps=self.input_seq_length
                )
                for _ in range(self.blocks-1)
            ]
        )

        # Dimension reduction layer 
        self.dimension_reduction = nn.AdaptiveAvgPool1d(1)

        # MLP head
        self.mlp_head = nn.Sequential(
            nn.Linear(self.node_count + self.input_seq_length*6, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )

    def forward(self, x, edge_index, imu_data):
        """
        Args:
            - x (PyTorch Float Tensor) Node features for T time periods, with shape (B, N_nodes, F_in, T).
            - edge_index (LongTensor): Edge indices with shape (2, E), where E is the number of edges, with values in {0, 1, ..., N_nodes - 1}.
                                       First row is the source nodes, second row is the target nodes.
            - imu_data (PyTorch Float Tensor) IMU data for T time periods, with shape (B, T, 6).
        Returns:
            - x (B, 6) - Odometry prediction
        """
        B, N_nodes, F_in, T = x.shape
        # Prepare edge index list for graph pooling
        edge_index_list = [edge_index for _ in range(T)]
        # loop over batch dimension
        pooled_x = torch.zeros((B, int(N_nodes*self.pooling_ratio), F_in, T))
        for i in range(B):
            # loop over time dimension
            for j in range(T):
                # in each iteration (1, N_nodes, F_in, T) -> (1, N_nodes_reduced, F_in, 1)
                pooled_x[i, :, :, j], edge_index_list[i], _, _, _, _ = self.graph_pooling(x[i, :, :, j], edge_index[i])
        x = pooled_x
        # (B, N_nodes, F_in, T) -> (B, N_nodes, nb_time_filter, T) 
        for astgcn_block in self.feature_extractor:
            x = astgcn_block(x, edge_index_list)
        # (B, N_nodes, nb_time_filter, T) -> (B, T, N_nodes, nb_time_filter)
        x = x.permute(0, 3, 1, 2)
        # (B, T, N_nodes, nb_time_filter) -> (B, 1, N_nodes, nb_time_filter) -> (B, N_nodes, nb_time_filter)
        x = x[:, 0, :, :].squeeze(1)
        # (B, N_nodes, nb_time_filter) -> (B, N_nodes)
        x = self.dimension_reduction(x).squeeze(-1)
        # (B, T, 6) -> (B, 6*T)
        imu_data = imu_data.reshape(B, 6*T)
        # (B, N_nodes -> (B, N_nodex + 6*T)
        x = torch.cat((x, imu_data), dim=1)
        # (B, N_nodex + 6*T) -> (B, 6)
        x = self.mlp_head(x)
        return x


if __name__ == "__main__":
    # Initialize model with desired parameters
    node_features = 3
    node_count = 5_000
    batch_size = 2
    input_seq_length = 4
    pooling_ratio = 0.05
    model = ASTGCN(node_features=node_features, input_seq_length=input_seq_length, node_count=node_count, blocks=3, pooling_ratio=pooling_ratio)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    # Example input tensor and edge index
    x = torch.rand((batch_size, node_count, node_features, input_seq_length))
    edge_index = torch.tensor([[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                            [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]],
                            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                            [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]]],
                            dtype=torch.long)
    imu_data = torch.rand((batch_size, input_seq_length, 6))
    # Forward pass
    output = model(x, edge_index, imu_data)
    print("Shape of out :", output.shape)
