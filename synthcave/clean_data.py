"""
clean_data.py

Reads LiDAR and IMU data from csv files, correct the csv headers, and save the data as numpy arrays.

Arguments:
----------
--source: Folder path to read data from.
--target: Folder path to write output data to.

Example Usage:
--------------
python synthcave/clean_data.py --source=data/1_raw --target=data/2_cleaning


The function loops through subfolders in the source folder, reads the data, corrects the headers, and saves the data in the target folder.

Author:
-------
Tim Bader
Date: January 21, 2024
"""
import numpy as np
import pandas as pd
import argparse
import math
import os


def read_lidar_csv(path: str) -> (np.array, np.array):
    """
    Read a csv file containing LiDAR data and return it as a numpy array.

    Arguments:
    ----------
    path: Path to the csv file.

    Returns:
    --------
    C_lidar: Numpy array containing the LiDAR data.
    ground_truth: Ground truth of the LiDAR data.
    """
    df = pd.read_csv(path, delimiter=';')
    # get c_lidar
    data_array = np.array([eval(x) for x in df['data']])
    # get ground truth
    ground_truth = get_ground_truth(df)
    return data_array, ground_truth


def get_diff(arr: np.array) -> np.array:
    """
    Calculate the difference between the elements of an array.

    Arguments:
    ----------
    arr: Array to calculate the difference from.

    Returns:
    --------
    diff: Array containing the difference between the elements of arr.
    """
    diff = np.zeros(len(arr))
    for i in range(1, len(arr)):
        diff[i] = arr[i] - arr[i-1]
    return diff


def to_spherical(x, y, z):
    """Converts a cartesian coordinate (x, y, z) into a spherical one (theta, phi)."""
    theta = math.atan2(math.sqrt(x * x + y * y), z)
    phi = math.atan2(y, x)
    return (math.degrees(theta), math.degrees(phi))


def get_ground_truth(df: pd.DataFrame) -> np.array:
    """
    Get the ground truth from a dataframe.

    Arguments:
    ----------
    df: Dataframe containing the ground truth.

    Returns:
    --------
    ground_truth: Ground truth of the dataframe.
    """
    # loop through rows and get ground truth
    
    pos_x = np.array([float(x) for x in df['posX']])
    pos_y = np.array([float(x) for x in df['posY']])
    pos_z = np.array([float(x) for x in df['posZ']])
    view_x = np.array([float(x) for x in df['viewX']])
    view_y = np.array([float(x) for x in df['viewY']])
    view_z = np.array([float(x) for x in df['viewZ']])
    theta = np.zeros(len(pos_x))
    phi = np.zeros(len(pos_x))
    for i in range(len(pos_x)):
        theta[i], phi[i] = to_spherical(view_x[i], view_z[i], view_y[i])
    # calculate ground truth
    diff_pos_x = get_diff(pos_x)
    diff_pos_y = get_diff(pos_y)
    diff_pos_z = get_diff(pos_z)
    diff_view_x = get_diff(view_x)
    diff_view_y = get_diff(view_y)
    diff_view_z = get_diff(view_z)
    diff_theta = get_diff(theta)
    diff_phi = get_diff(phi)
    # put ground truth together
    ground_truth = np.vstack((diff_pos_x, diff_pos_y, diff_pos_z, diff_view_x, diff_view_y, diff_view_z, diff_theta, diff_phi)).T
    return ground_truth


def read_imu_csv(path: str) -> (np.array, np.array):
    """
    Read a csv file containing IMU data and return it as a numpy array.

    Arguments:
    ----------
    path: Path to the csv file.

    Returns:
    --------
    imu_data: Numpy array containing the IMU data.
    ground_truth: Ground truth of the IMU data.
    """
    df = pd.read_csv(path, delimiter=';')
    # get imu_data
    imu_data = df[['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ']].values
    # get ground truth
    ground_truth = get_ground_truth(df)
    return imu_data, ground_truth


def correct_csv_header(path: str):
    """
    Correct the column headers in a csv file.

    Arguments:
    ----------
    path: Path to the csv file.
    """
    with open(path, 'r') as f:
        lines = f.readlines()
    lines[0] = lines[0].replace(',', ';')
    with open(path, 'w') as f:
        f.writelines(lines)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Clean data from the original dataset and save it as numpy arrays")
    parser.add_argument("--source", help="Folder path to read data from")
    parser.add_argument("--target", help="Folder path to write output data to")
    args = parser.parse_args()

    # loop through subfolders in source folder
    for root, dirs, _ in os.walk(args.source):
        for i, dir in enumerate(dirs):
            source_dir = os.path.join(root, dir)
            print(f"({i+1}/{len(dirs)}) Processing {dir}")
            # read data
            C_lidar1, GT_lidar1 = read_lidar_csv(os.path.join(source_dir, "lidar1.csv"))
            C_lidar2, GT_lidar2 = read_lidar_csv(os.path.join(source_dir, "lidar2.csv"))
            C_lidar3, GT_lidar3 = read_lidar_csv(os.path.join(source_dir, "lidar3.csv"))
            # correct headers (one comma is in there instead of ; -> fml)
            correct_csv_header(os.path.join(source_dir, "imu1.csv"))
            C_imu1, GT_imu1 = read_imu_csv(os.path.join(source_dir, "imu1.csv"))
            # create target folder
            target_dir = source_dir.replace(args.source, args.target, 1)
            os.makedirs(target_dir, exist_ok=True)
            # save data
            np.save(os.path.join(target_dir, "lidar1.npy"), C_lidar1)
            np.save(os.path.join(target_dir, "lidar1_gt.npy"), GT_lidar1)
            np.save(os.path.join(target_dir, "lidar2.npy"), C_lidar2)
            np.save(os.path.join(target_dir, "lidar2_gt.npy"), GT_lidar2)
            np.save(os.path.join(target_dir, "lidar3.npy"), C_lidar3)
            np.save(os.path.join(target_dir, "lidar3_gt.npy"), GT_lidar3)
            np.save(os.path.join(target_dir, "imu1.npy"), C_imu1)
            np.save(os.path.join(target_dir, "imu1_gt.npy"), GT_imu1)
    print("Done")
