import numpy as np

def distances_to_depth_image(C_lidar: np.array, lidar_range=100) -> np.array:
    """
    Creates a depth image list from LiDAR distances. A depth image is a 2D array 
    of uint16 values representing the distances returned by the LiDAR.

    Parameters:
    - C_lidar: 3D numpy array representing the NxHxV LiDAR distances as floats
    - lidar_range: int, maximum range of the LiDAR in meters

    Returns:
    - normed_depth_img_uint16: 3D numpy array representing the uint16 depth images
    """
    if lidar_range <= 0:
        raise ValueError("lidar_range must be greater than 0")
    if np.any(C_lidar < 0):
        raise ValueError("C_lidar must not contain negative values")
    if np.any(C_lidar > lidar_range):
        raise ValueError("C_lidar must not contain values greater than lidar_range")
    # Normalize the LiDAR distances to [0, 1]
    C_lidar_normed = C_lidar / lidar_range
    # Convert the normalized LiDAR distances to [0, 2**16 - 1]
    normed_depth_img_uint16 = (C_lidar_normed * (2**16 - 1)).astype(np.uint16)
    return normed_depth_img_uint16
