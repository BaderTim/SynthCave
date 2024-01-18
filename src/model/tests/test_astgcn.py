import torch
from model.ASTGCN.ASTGCN import ASTGCN


def test_astgcn_forward_call():
    node_features = 3
    node_count = 5_000
    batch_size = 2
    input_seq_length = 4
    pooling_ratio = 0.05
    model = ASTGCN(node_features=node_features, input_seq_length=input_seq_length, node_count=node_count, blocks=3, pooling_ratio=pooling_ratio)
    x = torch.rand((batch_size, node_count, node_features, input_seq_length))
    edge_index = torch.tensor([[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                            [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]],
                            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                            [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]]],
                            dtype=torch.long)
    imu_data = torch.rand((batch_size, input_seq_length, 6))
    output = model(x, edge_index, imu_data)
    assert output.shape == (batch_size, 6)
