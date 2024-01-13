import numpy as np
import random

def tf(C_lidar: np.array, p_occ=0.01):
    """
    Add techincal-fault-like-noise to an input sequence of LiDAR distance scans. When a technical fault occurs,
    the distance scan at that certain angle will be 0 for all time steps.

    Parameters:
    - p_occ: float, optional, probability of a distance scan angle being affected by a technical fault.

    Returns:
    - C_lidar_with_tf_noise: 3D numpy float array, shape (N, H, V), representing the input LiDAR data with added reflection-error-like noise.

    """
    C_lidar_with_tf_noise = C_lidar.copy()
    N, H, V = C_lidar.shape

    # Randomly decide whether to introduce a technical fault
    for h in range(H):
        for v in range(V):
            if random.random() < p_occ:
                # define time step from which the technical fault occurs
                tf_start_index = random.randint(0, N)
                # set all distance scans at that certain angle to 0
                C_lidar_with_tf_noise[tf_start_index:, h, v] = 0

    return C_lidar_with_tf_noise