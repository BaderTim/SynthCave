import numpy as np
import random

def ref(C_lidar: np.array, max_range=100, p_occ=0.2, occurence_range=100, time_length=50, hv_length=(20, 20)):
    """
    Add reflective-like noise to an input sequence of LiDAR distance scans. When a reflection is present, the distance scans
    tend to be either 0 or multiples of the actual distance to the object, eventually capped by the maximum range of the LiDAR.

    Parameters:
    - C_lidar: 3D numpy float array, shape (N, H, V), where N is the number of time steps, H the horizontal and V the vertical resolution.

    - max_range: int, optional, maximum range of the LiDAR.

    - p_occ: float, optional, probability of a reflection being present within occurence_range time steps.

    - occurence_range: int, optional, range in time steps within which a reflection can be present.

    - time_length: int, optional, maximum length in time steps of the reflection.

    - hv_length: tuple of ints, optional, maximum length in horizontal and vertical resolution of the reflection.

    Returns:
    - C_lidar_with_ref_noise: 3D numpy float array, shape (N, H, V), representing the input LiDAR data with added reflection-error-like noise.

    """
    C_lidar_with_ref_noise = C_lidar.copy()
    N, H, V = C_lidar.shape

    # For each occurence_range time steps, randomly decide whether to introduce a reflection
    for i in range(0, (N//occurence_range)*occurence_range, occurence_range):
        # Randomly decide whether to introduce a reflection
        if random.random() < p_occ:
            # Define the N,H,V-lengths of the reflection
            ref_n_length = random.randint(1, time_length)
            ref_h_length = random.randint(1, hv_length[0])
            ref_v_length = random.randint(1, hv_length[1])
            # Position the reflection within the 3D array
            ref_field_n_start_index = random.randint(i, i + occurence_range - ref_n_length)
            ref_field_n_end_index = ref_field_n_start_index + ref_n_length
            ref_field_h_start_index = random.randint(0, H - ref_h_length)
            ref_field_h_end_index = ref_field_h_start_index + ref_h_length
            ref_field_v_start_index = random.randint(0, V - ref_v_length)
            ref_field_v_end_index = ref_field_v_start_index + ref_v_length
            # Define the reflection: 
            # some values are 0, some are multiples of the actual distance to the object
            nhv_slice = C_lidar[ref_field_n_start_index:ref_field_n_end_index, 
                              ref_field_h_start_index:ref_field_h_end_index, 
                              ref_field_v_start_index:ref_field_v_end_index]
            nhv_ref = nhv_slice[nhv_slice != max_range] * np.random.randint(0, 4, size=nhv_ref[nhv_slice != max_range].shape)
            # cap the values by the maximum range of the LiDAR
            nhv_ref[nhv_ref > max_range] = max_range
            # Add the reflection to the 3D array
            C_lidar_with_ref_noise[ref_field_n_start_index:ref_field_n_end_index, 
                                   ref_field_h_start_index:ref_field_h_end_index, 
                                   ref_field_v_start_index:ref_field_v_end_index] = nhv_ref

    return C_lidar_with_ref_noise
