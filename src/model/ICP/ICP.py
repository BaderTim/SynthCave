# 3D ICP implementation
# paper: https://users.soe.ucsc.edu/~davis/papers/Mapping_IROS04/IROS04diebel.pdf

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

def icp(A, B, max_iterations=100, tolerance=1e-4):
    """
    Perform ICP to align point cloud A to point cloud B.

    Parameters:
    - A: numpy array, Nx3, source point cloud
    - B: numpy array, Nx3, target point cloud
    - max_iterations: int, maximum number of iterations
    - tolerance: float, convergence criterion

    Returns:
    - R: 3x3 rotation matrix
    - t: 1x3 translation vector
    - transformed_A: numpy array, Nx3, aligned source point cloud
    """

    for iteration in range(max_iterations):
        # Find the nearest neighbors using KDTree
        tree = KDTree(B)
        distances, indices = tree.query(A)

        # Extract corresponding points
        matched_points_A = A
        matched_points_B = B[indices]

        # Calculate the centroids of the matched points
        centroid_A = np.mean(matched_points_A, axis=0)
        centroid_B = np.mean(matched_points_B, axis=0)

        # Subtract the centroids to center the points
        centered_A = matched_points_A - centroid_A
        centered_B = matched_points_B - centroid_B

        # Compute the covariance matrix H
        H = np.dot(centered_A.T, centered_B)

        # Use Singular Value Decomposition to find the rotation matrix
        U, _, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        # Calculate the translation vector
        t = centroid_B - np.dot(R, centroid_A)

        # Transform the source point cloud using the calculated rotation and translation
        transformed_A = np.dot(A, R.T) + t

        # Check for convergence
        if np.sum(np.abs(transformed_A - A)) < tolerance:
            break

        A = transformed_A

    return R, t, transformed_A


if __name__ == "__main__":

    # Generate two example point clouds
    np.random.seed(42)
    A = np.random.randn(100, 3)  # Source point cloud
    R_true = Rotation.from_euler('xyz', [30, 45, 60], degrees=True).as_matrix()
    t_true = np.array([1, 2, 3])
    B = np.dot(A, R_true.T) + t_true + np.random.randn(100, 3) * 0.1  # Target point cloud with noise

    # Perform ICP
    R_estimate, t_estimate, aligned_A = icp(A, B)

    # Print the results
    print("True Rotation:")
    print(R_true)
    print("\nTrue Translation:")
    print(t_true)
    print("\nEstimated Rotation:")
    print(R_estimate)
    print("\nEstimated Translation:")
    print(t_estimate)
