import numpy as np
import random


def ref(C_lidar: np.array, max_range=100, p_occ=0.2, occurence_range=100, max_length=50, hv_length=(20, 20)):
    """
    Add reflective-like noise to an input sequence of LiDAR distance scans. When a reflection is present, the distance scans
    tend to be either 0 or multiples of the actual distance to the object, eventually capped by the maximum range of the LiDAR.

    Parameters:
    - C_lidar: 3D numpy float array, shape (N, H, V), where N is the number of time steps, H the horizontal and V the vertical resolution.

    - max_range: int, optional, maximum range of the LiDAR.

    - p_occ: float, optional, probability of a reflection being present within occurence_range time steps.

    - occurence_range: int, optional, range in time steps within which a reflection can be present.

    - max_length: int, optional, maximum length in time steps of the reflection.

    - hv_length: tuple of ints, optional, maximum length in horizontal and vertical resolution of the reflection.

    Returns:
    - C_lidar_with_ref_noise: 3D numpy float array, shape (N, H, V), representing the input LiDAR data with added reflection-error-like noise.

    """
    if p_occ == 0:
        return C_lidar
    if occurence_range > len(C_lidar):
        raise ValueError("occurence_range must be less than or equal to the length of C_lidar")
    if max_length > occurence_range:
        raise ValueError("max_length must be less than or equal to occurence_range")
    if hv_length[0] > C_lidar.shape[1]:
        raise ValueError("hv_length[0] must be less than or equal to the horizontal resolution of C_lidar")
    if hv_length[1] > C_lidar.shape[2]:
        raise ValueError("hv_length[1] must be less than or equal to the vertical resolution of C_lidar")
    C_lidar_with_ref_noise = C_lidar.copy()
    N, H, V = C_lidar.shape
    # For each occurence_range time steps, randomly decide whether to introduce a reflection
    for i in range(0, (N//occurence_range)*occurence_range, occurence_range):
        # Randomly decide whether to introduce a reflection
        if random.random() < p_occ:
            # Define the N,H,V-lengths of the reflection
            ref_n_length = random.randint(1, max_length)
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
            nhv_ref = nhv_slice[nhv_slice != max_range] * np.random.randint(0, 4, size=nhv_slice[nhv_slice != max_range].shape)
            # cap the values by the maximum range of the LiDAR
            nhv_ref[nhv_ref > max_range] = max_range
            # Add the reflection to the 3D array
            C_lidar_with_ref_noise[ref_field_n_start_index:ref_field_n_end_index, 
                                   ref_field_h_start_index:ref_field_h_end_index, 
                                   ref_field_v_start_index:ref_field_v_end_index] = nhv_ref
    return C_lidar_with_ref_noise


def tf(C_lidar: np.array, p_occ=0.01):
    """
    Add techincal-fault-like-noise to an input sequence of LiDAR distance scans. When a technical fault occurs,
    the distance scan at that certain angle will be 0 for all time steps.

    Parameters:
    - p_occ: float, optional, probability of a distance scan angle being affected by a technical fault.

    Returns:
    - C_lidar_with_tf_noise: 3D numpy float array, shape (N, H, V), representing the input LiDAR data with added reflection-error-like noise.

    """
    if p_occ == 0:
        return C_lidar
    C_lidar_with_tf_noise = C_lidar.copy()
    N, H, V = C_lidar.shape
    # Randomly decide whether to introduce a technical fault
    for h in range(H):
        for v in range(V):
            if random.random() < p_occ:
                # define time step from which the technical fault occurs
                tf_start_index = random.randint(0, N-1)
                # set all distance scans at that certain angle to 0
                C_lidar_with_tf_noise[tf_start_index:, h, v] = 0
    return C_lidar_with_tf_noise


def vol(C_lidar: np.array, max_range=100, p_occ=0.1, occurence_range=1000, max_length=200, hv_length=(60, 40), max_vol_impact=0.5):
    """
    Add different-volume-like noise to an input sequence of LiDAR distance scans. When a different volume is present, the distance scans
    tend to be larger than the actual distance to the object because in any other volume than air light travels slower through.

    Parameters:
    - C_lidar: 3D numpy float array, shape (N, H, V), where N is the number of time steps, H the horizontal and V the vertical resolution.

    - max_range: int, optional, maximum range of the LiDAR.

    - p_occ: float, optional, probability of a different volume being present within occurence_range time steps.

    - occurence_range: int, optional, range in time steps within which a different volume can be present.

    - max_length: int, optional, maximum length in time steps of the different volume.

    - hv_length: tuple of ints, optional, maximum length in horizontal and vertical resolution of the different volume.

    - max_vol_impact: float, optional, maximum impact of the different volume on the data.

    Returns:
    - C_lidar_with_vol_noise: 3D numpy float array, shape (N, H, V), representing the input LiDAR data with added different-volume-error-like noise.

    """
    if p_occ == 0:
        return C_lidar
    if occurence_range > len(C_lidar):
        raise ValueError("occurence_range must be less than or equal to the length of C_lidar")
    if max_length > occurence_range:
        raise ValueError("max_length must be less than or equal to occurence_range")
    if hv_length[0] > C_lidar.shape[1]:
        raise ValueError("hv_length[0] must be less than or equal to the horizontal resolution of C_lidar")
    if hv_length[1] > C_lidar.shape[2]:
        raise ValueError("hv_length[1] must be less than or equal to the vertical resolution of C_lidar")
    C_lidar_with_vol_noise = C_lidar.copy()
    N, H, V = C_lidar.shape
    # For each occurence_range time steps, randomly decide whether to introduce a volume
    for i in range(0, (N//occurence_range)*occurence_range, occurence_range):
        # Randomly decide whether to introduce a volume
        if random.random() < p_occ:
            # Define the N,H,V-lengths of the volume
            vol_n_length = random.randint(1, max_length)
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
