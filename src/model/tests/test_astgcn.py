import torch
from model.ASTGCN.ASTGCN import ASTGCN


def test_astgcn_forward_call():
    node_features = 3
    node_count = 19_200
    batch_size = 2
    K = 4
    model = ASTGCN(K=K)
    x = torch.rand((batch_size, node_count, node_features, K))
    edge_index = torch.tensor([[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                            [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]],
                            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                            [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]]],
                            dtype=torch.long)
    imu_data = torch.rand((batch_size, K, 6))
    output = model(x, edge_index, imu_data)
    assert output.shape == (batch_size, 6)
