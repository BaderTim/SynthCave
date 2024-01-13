import numpy as np

def cal(C_imu: np.array, std_dev=0.1):
    """
    Introduce calibration-error-like noise to an input sequence of IMU data.

    Parameters:
     - C_imu: 2D numpy array, shape (N, 6), where N is the number of time steps.
      Each inner list represents a set of IMU measurements for a specific time step, containing the following float values:
      [acceleration_X, acceleration_Y, acceleration_Z, rotation_rate_X, rotation_rate_Y, rotation_rate_Z]

    - std_dev: float, optional, magnitude of the calibration error impact on each measurement.

    Returns:
    - C_imu + noise_2d: 2D numpy array, shape (N, 6), representing the input IMU data with added calibration-error-like noise.

    """
    noise_1d = np.random.normal(0, std_dev, 6)
    # Repeat the same noise for each time step
    noise_2d = np.tile(noise_1d, (len(C_imu), 1))
    return C_imu + noise_2d
