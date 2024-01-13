import numpy as np
import random


def mag(C_imu: np.array, p_occ=0.1, max_length=20, impact_std_dev=0.1):
    """
    Add magnetic-field-like noise to an input sequence of IMU data.

    Parameters:
    - C_imu: 2D numpy array, shape (N, 6), where N is the number of time steps.
             Each inner list represents a set of IMU measurements for a specific time step, containing the following float values:
             [acceleration_X, acceleration_Y, acceleration_Z, rotation_rate_X, rotation_rate_Y, rotation_rate_Z]

    - p_occ: float, optional, probability of a magnetic field being present within 100 time steps.

    - max_length: int, optional, maximum length in time steps of the magnetic field.
    
    - impact_std_dev: float, optional, standard deviation of the impact of the magnetic field on the data.

    Returns:
    - C_imu + mag_noise_2d: 2D numpy array, shape (N, 6), representing the input IMU data with added calibration-error-like noise.

    """
    mag_noise_2d = np.zeros((len(C_imu), 6))

    # For each 100 time steps, randomly decide whether to introduce a magnetic field
    for i in range(0, (len(C_imu)//100)*100, 100):
        # Randomly decide whether to introduce a magnetic field
        if random.random() < p_occ:
            # Define the length of the magnetic field
            mag_field_length = random.randint(1, max_length)
            # Define the magnetic field
            x = np.linspace(0, mag_field_length, mag_field_length)
            gaussian = (1/(impact_std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mag_field_length // 2) / impact_std_dev)**2)
            mag_field = gaussian.reshape(-1, 1)
            # Randomly position the magnetic field within the current 100 time step window
            mag_noise_start_index = random.randint(i, i + 100 - mag_field_length)
            mag_noise_end_index = mag_noise_start_index + mag_field_length
            # Add the magnetic field to the 2D array, but only for rotational measurements
            mag_noise_2d[mag_noise_start_index:mag_noise_end_index, 3:6] += mag_field
    return C_imu + mag_noise_2d