import numpy as np
import random
import pytest
import re

from sdr.modules.lidar_sdr import ref, tf, vol

# Setting seed for reproducibility
np.random.seed(42)
random.seed(42)

def test_ref_no_reflection():
    # Test case assuming no change when p_occ is 0
    input_lidar_data = np.random.rand(5, 10, 10) 
    result = ref(input_lidar_data, p_occ=0.0, occurence_range=3, max_length=2, hv_length=(2, 2))
    assert np.array_equal(result, input_lidar_data)

def test_ref_with_reflection():
    # Test case assuming the parameters are properly set
    input_lidar_data = np.random.rand(5, 10, 10) 
    p_occ = 1
    result = ref(input_lidar_data, p_occ=p_occ, occurence_range=3, max_length=2, hv_length=(2, 2))
    # Check that the result is not equal to the input (due to reflection)
    assert not np.array_equal(result, input_lidar_data)

def test_ref_hv_length_h_error():
    # Test case assuming h from hv_length tuple is larger than lidar resolution and expects error
    input_lidar_data = np.random.rand(5, 10, 10) 
    with pytest.raises(ValueError, match=re.escape("hv_length[0] must be less than or equal to the horizontal resolution of C_lidar")):
        ref(input_lidar_data, hv_length=(15, 10), occurence_range=3, max_length=2)

def test_ref_hv_length_v_error():
    # Test case assuming v from hv_length tuple is larger than lidar resolution and expects error
    input_lidar_data = np.random.rand(5, 10, 10) 
    with pytest.raises(ValueError, match=re.escape("hv_length[1] must be less than or equal to the vertical resolution of C_lidar")):
        ref(input_lidar_data, hv_length=(10, 15), occurence_range=3, max_length=2)

def test_ref_occurrence_range_error():
    # Test case assuming time_length is larger than occurence_range and expects error
    input_lidar_data = np.random.rand(5, 10, 10) 
    with pytest.raises(ValueError, match="max_length must be less than or equal to occurence_range"):
        ref(input_lidar_data, occurence_range=3, max_length=4, hv_length=(2, 2))

def test_ref_input_length_error():
    # Test case assuming occurence_range is larger than input length and expects error
    input_lidar_data = np.random.rand(5, 10, 10) 
    with pytest.raises(ValueError, match="occurence_range must be less than or equal to the length of C_lidar"):
        ref(input_lidar_data, occurence_range=10 , max_length=2, hv_length=(2, 2))

def test_tf_no_fault():
    # Test case assuming no change when p_occ is 0
    input_lidar_data = np.random.rand(5, 10, 10) 
    result = tf(input_lidar_data, p_occ=0.0)
    assert np.array_equal(result, input_lidar_data)

def test_tf_last_index_zero():
    # Test case assuming the last index of C_lidar to be completely 0
    input_lidar_data = np.random.rand(5, 10, 10) 
    p_occ = 1.0  # Set p_occ to 1 for deterministic behavior in this test
    result = tf(input_lidar_data, p_occ=p_occ)
    # Check that the last index is completely 0
    assert np.all(result[-1] == 0)

def test_vol_no_volume():
    # Test case assuming no change when p_occ is 0
    input_lidar_data = np.random.rand(5, 10, 10) 
    result = vol(input_lidar_data, p_occ=0.0, occurence_range=3, max_length=2, hv_length=(2, 2))
    assert np.array_equal(result, input_lidar_data)

def test_vol_with_volume():
    # Test case assuming the parameters are properly set
    input_lidar_data = np.random.rand(5, 10, 10) 
    p_occ = 1
    result = vol(input_lidar_data, p_occ=p_occ, occurence_range=3, max_length=2, hv_length=(2, 2))
    # Check that the result is not equal to the input (due to volume)
    assert not np.array_equal(result, input_lidar_data)

def test_vol_hv_length_h_error():
    # Test case assuming h from hv_length tuple is larger than lidar resolution and expects error
    input_lidar_data = np.random.rand(5, 10, 10) 
    with pytest.raises(ValueError, match=re.escape("hv_length[0] must be less than or equal to the horizontal resolution of C_lidar")):
        vol(input_lidar_data, hv_length=(15, 10), occurence_range=3, max_length=2)

def test_vol_hv_length_v_error():
    # Test case assuming v from hv_length tuple is larger than lidar resolution and expects error
    input_lidar_data = np.random.rand(5, 10, 10) 
    with pytest.raises(ValueError, match=re.escape("hv_length[1] must be less than or equal to the vertical resolution of C_lidar")):
        vol(input_lidar_data, hv_length=(10, 15), occurence_range=3, max_length=2)

def test_vol_occurrence_range_error():
    # Test case assuming time_length is larger than occurence_range and expects error
    input_lidar_data = np.random.rand(5, 10, 10) 
    with pytest.raises(ValueError, match="max_length must be less than or equal to occurence_range"):
        vol(input_lidar_data, occurence_range=3, max_length=4, hv_length=(2, 2))

def test_vol_input_length_error():
    # Test case assuming occurence_range is larger than input length and expects error
    input_lidar_data = np.random.rand(5, 10, 10) 
    with pytest.raises(ValueError, match="occurence_range must be less than or equal to the length of C_lidar"):
        vol(input_lidar_data, occurence_range=10 , max_length=2, hv_length=(2, 2))
