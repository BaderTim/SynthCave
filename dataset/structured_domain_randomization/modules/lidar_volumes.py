import numpy as np
import random

def vol(C_lidar: np.array, max_range=100, p_occ=0.1, occurence_range=1000, time_length=200, hv_length=(60, 40), max_vol_impact=0.5):
    """
    Add different-volume-like noise to an input sequence of LiDAR distance scans. When a different volume is present, the distance scans
    tend to be larger than the actual distance to the object because in any other volume than air light travels slower through.

    Parameters:
    - C_lidar: 3D numpy float array, shape (N, H, V), where N is the number of time steps, H the horizontal and V the vertical resolution.

    - max_range: int, optional, maximum range of the LiDAR.

    - p_occ: float, optional, probability of a different volume being present within occurence_range time steps.

    - occurence_range: int, optional, range in time steps within which a different volume can be present.

    - time_length: int, optional, maximum length in time steps of the different volume.

    - hv_length: tuple of ints, optional, maximum length in horizontal and vertical resolution of the different volume.

    - max_vol_impact: float, optional, maximum impact of the different volume on the data.

    Returns:
    - C_lidar_with_vol_noise: 3D numpy float array, shape (N, H, V), representing the input LiDAR data with added different-volume-error-like noise.

    """
    C_lidar_with_vol_noise = C_lidar.copy()
    N, H, V = C_lidar.shape

    # For each occurence_range time steps, randomly decide whether to introduce a volume
    for i in range(0, (N//occurence_range)*occurence_range, occurence_range):
        # Randomly decide whether to introduce a volume
        if random.random() < p_occ:
            # Define the N,H,V-lengths of the volume
            vol_n_length = random.randint(1, time_length)
            vol_h_length = random.randint(1, hv_length[0])
            vol_v_length = random.randint(1, hv_length[1])
            # Position the volume within the 3D array
            vol_field_n_start_index = random.randint(i, i + occurence_range - vol_n_length)
            vol_field_n_end_index = vol_field_n_start_index + vol_n_length
            vol_field_h_start_index = random.randint(0, H - vol_h_length)
            vol_field_h_end_index = vol_field_h_start_index + vol_h_length
            vol_field_v_start_index = random.randint(0, V - vol_v_length)
            vol_field_v_end_index = vol_field_v_start_index + vol_v_length
            # Define the volume: 
            # all values are larger multiplied by a constant than the actual distance to the object
            C_lidar_with_vol_noise[vol_field_n_start_index:vol_field_n_end_index, 
                              vol_field_h_start_index:vol_field_h_end_index, 
                              vol_field_v_start_index:vol_field_v_end_index] *= random.uniform(1, 1+max_vol_impact)
    # cap the values by the maximum range of the LiDAR
    C_lidar_with_vol_noise[C_lidar_with_vol_noise > max_range] = max_range
    return C_lidar_with_vol_noise
