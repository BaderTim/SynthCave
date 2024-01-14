import numpy as np

def distances_to_point_cloud(C_lidar: np.array, lidar_h_angle: int, lidar_v_angle: int = 1) -> np.array:
    """
    Creates a point cloud list from LiDAR distances. A point cloud is a collection of points in 3D space,
    where the sensor is the origin of the coordinate system. 

    Parameters:
    - C_lidar: 3D numpy array representing the NxHxV LiDAR distances as floats
    - lidar_h_angle: int, horizontal angle of the LiDAR in degrees
    - lidar_v_angle: int, vertical angle of the LiDAR in degrees

    Returns:
    - point_clouds: 3D numpy array representing the point cloud like so: [[[x1, y1, z1], [x2, y2, z2], ...]]
    """
    if lidar_h_angle < 1 or lidar_h_angle > 360:
        raise ValueError("lidar_h_angle must be between 1 and 360 degrees")
    if lidar_v_angle < 1 or lidar_v_angle > 180:
        raise ValueError("lidar_v_angle must be between 1 and 180 degrees")
    N, H, V = C_lidar.shape
    point_clouds = np.zeros((N, H*V, 3))
    # calculate the rotation step for each angle
    h_rot_step = lidar_h_angle / H
    v_rot_step = lidar_v_angle / V
    # loop through time steps
    for n in range(N):
        # loop through horizontal angles
        for h in range(H):
            # calculate the rotation around the z-axis (azimuth)
            theta = np.deg2rad(h * h_rot_step - lidar_h_angle / 2)
            # loop through vertical angles
            for v in range(V):
                # calculate the rotation around the y-axis
                phi = np.deg2rad(90 - lidar_v_angle / 2 + v * v_rot_step)
                # calculate x, y, z coordinates for the current point
                x = C_lidar[n, h, v] * np.sin(phi) * np.cos(theta)
                y = C_lidar[n, h, v] * np.sin(phi) * np.sin(theta)
                z = C_lidar[n, h, v] * np.cos(phi)
                # add the point to the point cloud
                point_clouds[n, h*V + v, :] = [x, y, z]
    return point_clouds
