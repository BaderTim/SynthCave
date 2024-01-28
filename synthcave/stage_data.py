"""
stage_data.py

This script applies transformations to LiDAR data to generate different representations:
depth images, point clouds, and graphs.

Example Usage:
--------------
Assuming your LiDAR data is stored in CSV files, with corresponding ranges, horizontal 
angles, and vertical angles, you can use the following command to transform the data:

$ python synthcave/stage_data.py --source=data/3_derived --target=data/4_staging --devices=data/devices.json --train=10 --val=4 --test=6

Required Arguments:
-------------------
--source: Folder path to read data from.
--target: Folder path to write output data to.
--devices: JSON file containing the device parameters.
--train: Define amount of training samples per sequence.
--val: Define amount of validation samples per sequence.
--test: Define amount of test samples per sequence.

Author:
-------
Tim Bader
Date: January 21, 2024
"""
import numpy as np
import argparse
import json
import shutil
import os

from transform.modules.depth_image_transform import distances_to_depth_image
from transform.modules.point_cloud_transform import distances_to_point_cloud
from transform.modules.graph_transform import distances_to_graph


def sync_imu_with_lidar(C_lidar: np.array, C_imu: np.array, imu: dict, lidar: dict) -> np.array:
    """
    Synchronizes the frequency of the IMU data with the frequency of the LiDAR data.

    Parameters:
    - C_lidar: 3D numpy float array, shape (N, H, W), where N is the number of time steps and M is the number of LiDAR measurements per time step.
    - C_imu: 2D numpy float array, shape (N, 6), where N is the number of time steps.
             Each inner list represents a set of IMU measurements for a specific time step, containing the following float values:
             [acceleration_X, acceleration_Y, acceleration_Z, rotation_rate_X, rotation_rate_Y, rotation_rate_Z]
    - imu: dict, while having other parameters, also contains frequency as int
    - lidar: dict, while having other parameters, also contains frequency as int

    Returns:
    - C_imu_downsampled: 2D numpy float array, shape (N, 6), where N is the number of time steps.
                         Each inner list represents a set of IMU measurements for a specific time step, containing the following float values:
                         [acceleration_X, acceleration_Y, acceleration_Z, rotation_rate_X, rotation_rate_Y, rotation_rate_Z]
    """
    imu_frequency = imu["frequency"]
    lidar_frequency = lidar["frequency"]
    if imu_frequency == lidar_frequency:
        return C_imu
    elif imu_frequency > lidar_frequency:
        # IMU frequency is higher than LiDAR frequency
        # -> IMU data needs to be downsampled by addition
        downsample_factor = imu_frequency // lidar_frequency
        # special case: when upsampling lidar at same factor, one imu entry is missing
        # -> add one imu entry at the end
        if (len(C_imu) < len(C_lidar) * downsample_factor):
            C_imu = np.vstack((C_imu, C_imu[-1]))
        C_imu_downsampled = np.zeros((len(C_imu)//downsample_factor, 6))
        # optionally truncate C_imu to be divisible by downsample_factor
        for i in range(0, len(C_imu) - len(C_imu) % downsample_factor, downsample_factor):
            C_imu_downsampled[i//downsample_factor] = np.sum(C_imu[i:i+downsample_factor], axis=0)
        if len(C_imu_downsampled) != len(C_lidar):
            print("Downsampled IMU data does not match LiDAR data")
        return C_imu_downsampled
    

def sort_recordings(dirs: list) -> list:
    """
    Sorts the recordings in the given list by their number.

    Parameters:
    - dirs: list of strings, where each string represents a recording folder, e.g. "curvy_path_downwards_1", "curvy_path_downwards_20", ...

    Returns:
    - sorted_dirs: list of strings, sorted by a) the section name and b) the number
    """
    def custom_sort_key(directory):
        section_name, number = directory.rsplit('_', 1)
        # Return a tuple to use as the sorting key
        return (section_name, int(number))
    # Sort the directories using the custom sorting key
    sorted_dirs = sorted(dirs, key=custom_sort_key)
    return sorted_dirs


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Transformations to receive the representations mentioned in the SynthCave paper")
    parser.add_argument("--source", help="Folder path to read data from")
    parser.add_argument("--target", help="Folder path to write output data to")
    parser.add_argument("--devices", help="JSON file containing the device parameters")
    parser.add_argument("--train", help="Define amount of training samples per sequence")
    parser.add_argument("--val", help="Define amount of validation samples per sequence")
    parser.add_argument("--test", help="Define amount of test samples per sequence")
    args = parser.parse_args()
    # Load data
    devices = json.load(open(args.devices, "r"))
    lidar1 = devices["lidar1"]
    lidar2 = devices["lidar2"]
    lidar3 = devices["lidar3"]
    imu1 = devices["imu1"]
    # create target folders
    os.makedirs(os.path.join(args.target, "lidar1/graph/train"), exist_ok=True)
    os.makedirs(os.path.join(args.target, "lidar1/graph/val"), exist_ok=True)
    os.makedirs(os.path.join(args.target, "lidar1/graph/test"), exist_ok=True)
    os.makedirs(os.path.join(args.target, "lidar2/graph/train"), exist_ok=True)
    os.makedirs(os.path.join(args.target, "lidar2/graph/val"), exist_ok=True)
    os.makedirs(os.path.join(args.target, "lidar2/graph/test"), exist_ok=True)
    os.makedirs(os.path.join(args.target, "lidar3/graph/train"), exist_ok=True)
    os.makedirs(os.path.join(args.target, "lidar3/graph/val"), exist_ok=True)
    os.makedirs(os.path.join(args.target, "lidar3/graph/test"), exist_ok=True)

    os.makedirs(os.path.join(args.target, "lidar1/depth_image/train"), exist_ok=True)
    os.makedirs(os.path.join(args.target, "lidar1/depth_image/val"), exist_ok=True)
    os.makedirs(os.path.join(args.target, "lidar1/depth_image/test"), exist_ok=True)
    os.makedirs(os.path.join(args.target, "lidar2/depth_image/train"), exist_ok=True)
    os.makedirs(os.path.join(args.target, "lidar2/depth_image/val"), exist_ok=True)
    os.makedirs(os.path.join(args.target, "lidar2/depth_image/test"), exist_ok=True)
    os.makedirs(os.path.join(args.target, "lidar3/depth_image/train"), exist_ok=True)
    os.makedirs(os.path.join(args.target, "lidar3/depth_image/val"), exist_ok=True)
    os.makedirs(os.path.join(args.target, "lidar3/depth_image/test"), exist_ok=True)

    os.makedirs(os.path.join(args.target, "lidar1/point_cloud/train"), exist_ok=True)
    os.makedirs(os.path.join(args.target, "lidar1/point_cloud/val"), exist_ok=True)
    os.makedirs(os.path.join(args.target, "lidar1/point_cloud/test"), exist_ok=True)
    os.makedirs(os.path.join(args.target, "lidar2/point_cloud/train"), exist_ok=True)
    os.makedirs(os.path.join(args.target, "lidar2/point_cloud/val"), exist_ok=True)
    os.makedirs(os.path.join(args.target, "lidar2/point_cloud/test"), exist_ok=True)
    os.makedirs(os.path.join(args.target, "lidar3/point_cloud/train"), exist_ok=True)
    os.makedirs(os.path.join(args.target, "lidar3/point_cloud/val"), exist_ok=True)
    os.makedirs(os.path.join(args.target, "lidar3/point_cloud/test"), exist_ok=True)

    section = {}
    indexes = {"total_train_index": 0, "total_val_index": 0, "total_test_index": 0}

    # loop through subfolders in source folder
    for root, dirs, _ in os.walk(args.source):
        dirs = sort_recordings(dirs)
        for i, dir in enumerate(dirs):
            source_dir = os.path.join(root, dir)
            print(f"({i+1}/{len(dirs)}) Processing {dir}")
            section_name, number = dir.rsplit('_', 1)
            # determine path end (train, val, test)
            section_name, number = dir.rsplit('_', 1)
            if section_name not in section:
                section[section_name] = {"train": 0, "val": 0, "test": 0}
            
            if section[section_name]["train"] < int(args.train):
                section[section_name]["train"] += 1
                index = indexes["total_train_index"]
                indexes["total_train_index"] += 1
                path_end = "train"
            elif section[section_name]["val"] < int(args.val):
                index = indexes["total_val_index"]
                section[section_name]["val"] += 1
                indexes["total_val_index"] += 1
                path_end = "val"
            elif section[section_name]["test"] < int(args.test):
                index = indexes["total_test_index"]
                section[section_name]["test"] += 1
                indexes["total_test_index"] += 1
                path_end = "test"

            # read data
            C_imu1 = np.load(os.path.join(source_dir, "imu1.npy"))
            C_lidar1 = np.load(os.path.join(source_dir, "lidar1.npy"))
            C_lidar1_imu1 = sync_imu_with_lidar(C_lidar1, C_imu1, imu1, lidar1)
            C_lidar1_gt = np.load(os.path.join(source_dir, "lidar1_gt.npy"))
            C_lidar2 = np.load(os.path.join(source_dir, "lidar2.npy"))
            C_lidar2_imu1 = sync_imu_with_lidar(C_lidar1, C_imu1, imu1, lidar2)
            C_lidar2_gt = np.load(os.path.join(source_dir, "lidar2_gt.npy"))
            C_lidar3 = np.load(os.path.join(source_dir, "lidar3.npy"))
            C_lidar3_imu1 = sync_imu_with_lidar(C_lidar1, C_imu1, imu1, lidar3)
            C_lidar3_gt = np.load(os.path.join(source_dir, "lidar3_gt.npy"))

            # build graphs
            C_lidar1_graph, C_lidar1_edges = distances_to_graph(
                C_lidar=C_lidar1,
                lidar_h_angle=lidar1["horizontal_fov"],
                lidar_v_angle=lidar1["vertical_fov"],
                lidar_range=lidar1["range"]
            )
            np.save(os.path.join(args.target, f"lidar1/graph/{path_end}/{index}_{section_name}_graph.npy"), C_lidar1_graph)
            np.save(os.path.join(args.target, f"lidar1/graph/{path_end}/{index}_{section_name}_edges.npy"), C_lidar1_edges)
            np.save(os.path.join(args.target, f"lidar1/graph/{path_end}/{index}_{section_name}_imu.npy"), C_lidar1_imu1)
            shutil.copy(os.path.join(source_dir, f"lidar1_gt.npy"), os.path.join(args.target, f"lidar1/graph/{path_end}/{index}_{section_name}_gt.npy"))
            C_lidar2_graph, C_lidar2_edges = distances_to_graph(
                C_lidar=C_lidar2,
                lidar_h_angle=lidar2["horizontal_fov"],
                lidar_v_angle=lidar2["vertical_fov"],
                lidar_range=lidar2["range"]
            )
            np.save(os.path.join(args.target, f"lidar2/graph/{path_end}/{index}_{section_name}_graph.npy"), C_lidar2_graph)
            np.save(os.path.join(args.target, f"lidar2/graph/{path_end}/{index}_{section_name}_edges.npy"), C_lidar2_edges)
            np.save(os.path.join(args.target, f"lidar2/graph/{path_end}/{index}_{section_name}_imu.npy"), C_lidar2_imu1)
            shutil.copy(os.path.join(source_dir, f"lidar2_gt.npy"), os.path.join(args.target, f"lidar2/graph/{path_end}/{index}_{section_name}_gt.npy"))
            C_lidar3_graph, C_lidar3_edges = distances_to_graph(
                C_lidar=C_lidar3,
                lidar_h_angle=lidar3["horizontal_fov"],
                lidar_v_angle=lidar3["vertical_fov"],
                lidar_range=lidar3["range"]
            )
            np.save(os.path.join(args.target, f"lidar3/graph/{path_end}/{index}_{section_name}_graph.npy"), C_lidar3_graph)
            np.save(os.path.join(args.target, f"lidar3/graph/{path_end}/{index}_{section_name}_edges.npy"), C_lidar3_edges)
            np.save(os.path.join(args.target, f"lidar3/graph/{path_end}/{index}_{section_name}_imu.npy"), C_lidar3_imu1)
            shutil.copy(os.path.join(source_dir, f"lidar3_gt.npy"), os.path.join(args.target, f"lidar3/graph/{path_end}/{index}_{section_name}_gt.npy"))

            # build point clouds
            C_lidar1_point_cloud = distances_to_point_cloud(
                C_lidar=C_lidar1,
                lidar_h_angle=lidar1["horizontal_fov"],
                lidar_v_angle=lidar1["vertical_fov"],
                lidar_range=lidar1["range"]
            )
            np.save(os.path.join(args.target, f"lidar1/point_cloud/{path_end}/{index}_{section_name}_pc.npy"), C_lidar1_point_cloud)
            np.save(os.path.join(args.target, f"lidar1/point_cloud/{path_end}/{index}_{section_name}_imu.npy"), C_lidar1_imu1)
            shutil.copy(os.path.join(source_dir, f"lidar1_gt.npy"), os.path.join(args.target, f"lidar1/point_cloud/{path_end}/{index}_{section_name}_gt.npy"))
            C_lidar2_point_cloud = distances_to_point_cloud(
                C_lidar=C_lidar2,
                lidar_h_angle=lidar2["horizontal_fov"],
                lidar_v_angle=lidar2["vertical_fov"],
                lidar_range=lidar2["range"]
            )
            np.save(os.path.join(args.target, f"lidar2/point_cloud/{path_end}/{index}_{section_name}_pc.npy"), C_lidar2_point_cloud)
            np.save(os.path.join(args.target, f"lidar2/point_cloud/{path_end}/{index}_{section_name}_imu.npy"), C_lidar2_imu1)
            shutil.copy(os.path.join(source_dir, f"lidar2_gt.npy"), os.path.join(args.target, f"lidar2/point_cloud/{path_end}/{index}_{section_name}_gt.npy"))
            C_lidar3_point_cloud = distances_to_point_cloud(
                C_lidar=C_lidar3,
                lidar_h_angle=lidar3["horizontal_fov"],
                lidar_v_angle=lidar3["vertical_fov"],
                lidar_range=lidar3["range"]
            )
            np.save(os.path.join(args.target, f"lidar3/point_cloud/{path_end}/{index}_{section_name}_pc.npy"), C_lidar3_point_cloud)
            np.save(os.path.join(args.target, f"lidar3/point_cloud/{path_end}/{index}_{section_name}_imu.npy"), C_lidar3_imu1)
            shutil.copy(os.path.join(source_dir, f"lidar3_gt.npy"), os.path.join(args.target, f"lidar3/point_cloud/{path_end}/{index}_{section_name}_gt.npy"))

            # build depth images
            C_lidar1_depth_image = distances_to_depth_image(
                C_lidar=C_lidar1,
                lidar_range=lidar1["range"]
            )
            np.save(os.path.join(args.target, f"lidar1/depth_image/{path_end}/{index}_{section_name}_img.npy"), C_lidar1_depth_image)
            np.save(os.path.join(args.target, f"lidar1/depth_image/{path_end}/{index}_{section_name}_imu.npy"), C_lidar1_imu1)
            shutil.copy(os.path.join(source_dir, f"lidar1_gt.npy"), os.path.join(args.target, f"lidar1/depth_image/{path_end}/{index}_{section_name}_gt.npy"))
            C_lidar2_depth_image = distances_to_depth_image(
                C_lidar=C_lidar2,
                lidar_range=lidar2["range"]
            )
            np.save(os.path.join(args.target, f"lidar2/depth_image/{path_end}/{index}_{section_name}_img.npy"), C_lidar2_depth_image)
            np.save(os.path.join(args.target, f"lidar2/depth_image/{path_end}/{index}_{section_name}_imu.npy"), C_lidar2_imu1)
            shutil.copy(os.path.join(source_dir, f"lidar2_gt.npy"), os.path.join(args.target, f"lidar2/depth_image/{path_end}/{index}_{section_name}_gt.npy"))
            C_lidar3_depth_image = distances_to_depth_image(
                C_lidar=C_lidar3,
                lidar_range=lidar3["range"]
            )
            np.save(os.path.join(args.target, f"lidar3/depth_image/{path_end}/{index}_{section_name}_img.npy"), C_lidar3_depth_image)
            np.save(os.path.join(args.target, f"lidar3/depth_image/{path_end}/{index}_{section_name}_imu.npy"), C_lidar3_imu1)
            shutil.copy(os.path.join(source_dir, f"lidar3_gt.npy"), os.path.join(args.target, f"lidar3/depth_image/{path_end}/{index}_{section_name}_gt.npy"))
