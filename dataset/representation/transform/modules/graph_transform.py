import numpy as np

def distances_to_graph(C_lidar: np.array, lidar_h_angle: int, lidar_v_angle: int) -> tuple:
    """
    Creates a temporal graph from LiDAR distances. The graph is a collection of nodes and edges.
    Distances and angles are features of the nodes, edges are connections between nodes.

    Parameters:
    - C_lidar_n: 3D numpy array representing the NxHxV LiDAR distances as floats
    - lidar_h_angle: int, horizontal angle of the LiDAR in degrees
    - lidar_v_angle: int, vertical angle of the LiDAR in degrees

    Returns:
    - (graphs, edges): tuple of numpy arrays with a graph (time_step, nodes, features) where features
                      are (distance, theta, phi) and non-directed edges (time_step, source, target)
    """
    if lidar_h_angle < 1 or lidar_h_angle > 360:
        raise ValueError("lidar_h_angle must be between 1 and 360 degrees")
    if lidar_v_angle < 1 or lidar_v_angle > 180:
        raise ValueError("lidar_v_angle must be between 1 and 180 degrees")
    N, H, V = C_lidar.shape
    edge_count = (H-1)*V + H*(V-1)
    graphs = np.zeros((N, H*V, 3))
    edges = np.zeros((N, edge_count, 2))
    # loop through time steps
    for n in range(N):
        # calculate the rotation step for each angle
        h_rot_step = lidar_h_angle / H
        v_rot_step = lidar_v_angle / V
        edge_index = 0
        # loop through horizontal angles
        for h in range(H):
            # calculate the rotation around the z-axis (azimuth)
            theta = np.deg2rad(h * h_rot_step - lidar_h_angle / 2)
            # loop through vertical angles
            for v in range(V):
                # calculate the rotation around the y-axis
                phi = np.deg2rad(90 - lidar_v_angle / 2 + v * v_rot_step)
                node_index = h*V + v
                # add the feature to the node
                graphs[n, node_index, :] = [C_lidar[n, h, v], theta, phi]
                # add horizontal edge towards right if not last column
                if h < H - 1:
                    edges[n, edge_index, :] = [node_index, node_index + V]
                    edge_index += 1
                # add vertical edge towards bottom if not last row
                if v < V - 1:
                    edges[n, edge_index, :] = [node_index, node_index + 1]
                    edge_index += 1
    return (graphs, edges)
