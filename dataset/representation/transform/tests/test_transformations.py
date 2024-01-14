import numpy as np
import pytest

from transform.modules.depth_image_transform import distances_to_depth_image
from transform.modules.point_cloud_transform import distances_to_point_cloud
from transform.modules.graph_transform import distances_to_graph

def test_distances_to_depth_image_main_case():
    # Test main case with typical input
    input_lidar_data = np.array([[[0.0, 1.0], [2.0, 2.0]]]) 
    result = distances_to_depth_image(input_lidar_data, lidar_range=2)
    expected_result = np.array([[[0, (2**16 - 1)//2], [2**16 - 1, 2**16 - 1]]])
    assert np.array_equal(result, expected_result)

def test_distances_to_depth_image_edge_case_max_values():
    # Test edge case with maximum float values
    input_lidar_data = np.array([[[np.finfo(float).max, 0.0], [0.0, np.finfo(float).max]]]) 
    result = distances_to_depth_image(input_lidar_data, lidar_range=np.finfo(float).max)
    expected_result = np.array([[[2**16 - 1, 0], [0, 2**16 - 1]]])
    assert np.array_equal(result, expected_result)

def test_distances_to_depth_image_invalid_lidar_range():
    # Test case with an invalid lidar_range (<= 0)
    input_lidar_data = np.random.rand(5, 10, 10)  # Adjust dimensions as needed
    invalid_lidar_range = 0
    with pytest.raises(ValueError, match="lidar_range must be greater than 0"):
        distances_to_depth_image(input_lidar_data, lidar_range=invalid_lidar_range)

def test_distances_to_depth_image_negative_values_in_C_lidar():
    # Test case with negative values in C_lidar
    input_lidar_data = np.random.uniform(-10, 10, size=(5, 10, 10))  # Adjust dimensions as needed
    with pytest.raises(ValueError, match="C_lidar must not contain negative values"):
        distances_to_depth_image(input_lidar_data)

def test_distances_to_depth_image_values_above_lidar_range():
    # Test case with values in C_lidar above lidar_range
    input_lidar_data = np.random.uniform(0, 150, size=(5, 10, 10))  # Adjust dimensions as needed
    invalid_lidar_range = 100
    with pytest.raises(ValueError, match="C_lidar must not contain values greater than lidar_range"):
        distances_to_depth_image(input_lidar_data, lidar_range=invalid_lidar_range)

def test_distances_to_point_cloud_main_case():
    # Test main case with typical input
    input_lidar_data = np.array([[[1.0, 1.0], [1.0, 1.0]]])
    lidar_h_angle = 90
    lidar_v_angle = 40
    result = distances_to_point_cloud(input_lidar_data, lidar_h_angle, lidar_v_angle)
    # disable scientific notation
    expected_result = np.array([[
        [0.66, -0.66, 0.34],
        [0.71, -0.71, 0],
        [0.94, 0, 0.34],
        [1, 0, 0]
    ]])
    np.testing.assert_allclose(result, expected_result, rtol=0, atol=0.01)

def test_distances_to_point_cloud_edge_case_invalid_h_angle():
    # Test edge case with an invalid horizontal angle
    input_lidar_data = np.array([[[1.0, 2.0], [3.0, 4.0]]]) 
    lidar_h_angle = 361  # Invalid horizontal angle
    lidar_v_angle = 45
    with pytest.raises(ValueError, match="lidar_h_angle must be between 1 and 360 degrees"):
        distances_to_point_cloud(input_lidar_data, lidar_h_angle, lidar_v_angle)

def test_distances_to_point_cloud_edge_case_invalid_v_angle():
    # Test edge case with an invalid vertical angle
    input_lidar_data = np.array([[[1.0, 2.0], [3.0, 4.0]]]) 
    lidar_h_angle = 90
    lidar_v_angle = 181  # Invalid vertical angle
    with pytest.raises(ValueError, match="lidar_v_angle must be between 1 and 180 degrees"):
        distances_to_point_cloud(input_lidar_data, lidar_h_angle, lidar_v_angle)

def test_distances_to_point_cloud_edge_case_h_angle_below_1():
    # Test edge case with horizontal angle below 1
    input_lidar_data = np.array([[[1.0, 2.0], [3.0, 4.0]]]) 
    lidar_h_angle = 0
    lidar_v_angle = 45
    with pytest.raises(ValueError, match="lidar_h_angle must be between 1 and 360 degrees"):
        distances_to_point_cloud(input_lidar_data, lidar_h_angle, lidar_v_angle)

def test_distances_to_point_cloud_edge_case_v_angle_below_1():
    # Test edge case with vertical angle below 1
    input_lidar_data = np.array([[[1.0, 2.0], [3.0, 4.0]]]) 
    lidar_h_angle = 90
    lidar_v_angle = 0
    with pytest.raises(ValueError, match="lidar_v_angle must be between 1 and 180 degrees"):
        distances_to_point_cloud(input_lidar_data, lidar_h_angle, lidar_v_angle)

def test_distances_to_graph_main_functionality():
    # Test main case with typical input
    input_lidar_data = np.array([[[1.0, 2.0, 3.0],
                                  [4.0, 5.0, 6.0],
                                  [7.0, 8.0, 9.0]]]) 
    lidar_h_angle = 90
    lidar_v_angle = 45
    graphs, edges = distances_to_graph(input_lidar_data, lidar_h_angle, lidar_v_angle)
    # Check the shape of the resulting graph and edges
    assert graphs.shape == (1, input_lidar_data.shape[1] * input_lidar_data.shape[2], 3)
    assert edges.shape == (1, (input_lidar_data.shape[1] - 1) * input_lidar_data.shape[2] +
                           input_lidar_data.shape[1] * (input_lidar_data.shape[2] - 1), 2)
    # Check the correctness of node features
    assert np.array_equal(graphs[0, :, 0], input_lidar_data[0].flatten())
    # Check the correctness of edges
    assert np.all(edges >= 0)  # Ensure all indices are non-negative
    assert np.all(edges < graphs.shape[1])  # Ensure all indices are within the range of nodes

def test_distances_to_graph_edge_case_invalid_h_angle():
    # Test edge case with an invalid horizontal angle
    input_lidar_data = np.array([[[0.0, 1.0], [2.0, 3.0]]]) 
    lidar_h_angle = 361  # Invalid horizontal angle
    lidar_v_angle = 45
    with pytest.raises(ValueError, match="lidar_h_angle must be between 1 and 360 degrees"):
        distances_to_graph(input_lidar_data, lidar_h_angle, lidar_v_angle)

def test_distances_to_graph_edge_case_invalid_v_angle():
    # Test edge case with an invalid vertical angle
    input_lidar_data = np.array([[[0.0, 1.0], [2.0, 3.0]]]) 
    lidar_h_angle = 90
    lidar_v_angle = 181  # Invalid vertical angle
    with pytest.raises(ValueError, match="lidar_v_angle must be between 1 and 180 degrees"):
        distances_to_graph(input_lidar_data, lidar_h_angle, lidar_v_angle)

def test_distances_to_graph_edge_case_h_angle_below_1():
    # Test edge case with horizontal angle below 1
    input_lidar_data = np.array([[[0.0, 1.0], [2.0, 3.0]]]) 
    lidar_h_angle = 0
    lidar_v_angle = 45
    with pytest.raises(ValueError, match="lidar_h_angle must be between 1 and 360 degrees"):
        distances_to_graph(input_lidar_data, lidar_h_angle, lidar_v_angle)

def test_distances_to_graph_edge_case_v_angle_below_1():
    # Test edge case with vertical angle below 1
    input_lidar_data = np.array([[[0.0, 1.0], [2.0, 3.0]]]) 
    lidar_h_angle = 90
    lidar_v_angle = 0
    with pytest.raises(ValueError, match="lidar_v_angle must be between 1 and 180 degrees"):
        distances_to_graph(input_lidar_data, lidar_h_angle, lidar_v_angle)
