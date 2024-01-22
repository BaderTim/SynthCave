import numpy as np
import random
import pytest

from sdr.modules.imu_sdr import cal, mag

# Setting seed for reproducibility
np.random.seed(42)
random.seed(42)


def test_cal_no_calibration_error():
    # Test case assuming std_dev is 0, so there should be no calibration error introduced
    input_imu_data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                               [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]])
    result = cal(input_imu_data, std_dev=0.0)
    assert np.array_equal(result, input_imu_data)

def test_cal_with_calibration_error():
    # Test case with a non-zero std_dev, ensuring calibration error is introduced
    input_imu_data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                               [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]])
    std_dev = 0.1
    result = cal(input_imu_data, std_dev=std_dev)

    # Check that the result is not equal to the input (due to calibration error)
    assert not np.array_equal(result, input_imu_data)

def test_mag_no_magnetic_field():
    # Test case assuming p_occ is 0, so there should be no magnetic field introduced
    input_imu_data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                               [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]])
    result = mag(input_imu_data, occurence_range=3, max_length=2, p_occ=0.0)
    assert np.array_equal(result, input_imu_data)

def test_mag_with_magnetic_field():
    # Test case with a non-zero p_occ, ensuring a magnetic field is introduced
    input_imu_data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                               [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                               [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                               [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]])
    p_occ = 1
    result = mag(input_imu_data, p_occ=p_occ, occurence_range=3, max_length=2)

    # Check that the result is not equal to the input (due to magnetic field)
    assert not np.array_equal(result, input_imu_data)

    # Check that the magnetic field has been introduced
    assert np.any(result[:, 3:6] != input_imu_data[:, 3:6])

def test_mag_occurrence_range_error():
    # Test case where occurence_range is greater than the length of C_imu
    input_imu_data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    with pytest.raises(ValueError, match="occurence_range must be less than or equal to the length of C_imu"):
        mag(input_imu_data, occurence_range=2)

def test_mag_max_length_error():
    # Test case where max_length is greater than occurence_range
    input_imu_data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    with pytest.raises(ValueError, match="max_length must be less than or equal to occurence_range"):
        mag(input_imu_data, max_length=3, occurence_range=1)
