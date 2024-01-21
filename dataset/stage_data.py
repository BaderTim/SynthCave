"""
transform.py

This script applies transformations to LiDAR data to generate different representations, 
including depth images, point clouds, and graphs.

Example Usage:
--------------
Assuming your LiDAR data is stored in CSV files, with corresponding ranges, horizontal 
angles, and vertical angles, you can use the following command to transform the data:

$ python transform.py --lidar_data lidar1.csv lidar2.csv --lidar_ranges 100 150 --lidar_h_angles 90 45 --lidar_v_angles 30 60 --rep depth_image --output output_folder

Required Arguments:
-------------------
--lidar_data: Array of csv file paths containing LiDAR data.
--lidar_ranges: Array of LiDAR ranges in meters.
--lidar_h_angles: Array of LiDAR horizontal angles in degrees.
--lidar_v_angles: Array of LiDAR vertical angles in degrees.
--rep: Representation to apply. One of "depth_image", "point_cloud", "graph".
--output: Folder path to write output data to.

Author:
-------
Tim Bader
Date: January 14, 2024
"""
import numpy as np
import argparse
import csv

from transform.modules.depth_image_transform import distances_to_depth_image
from transform.modules.point_cloud_transform import distances_to_point_cloud
from transform.modules.graph_transform import distances_to_graph


def depth_image(C_lidars: list[np.array], lidar_ranges: np.array, lidar_h_angles: np.array, lidar_v_angles: np.array) -> list[np.array]:
    transformed_C_lidars = []
    for i in range(len(C_lidars)):
        transformed_C_lidars.append(distances_to_depth_image(
            C_lidar=C_lidars[i],
            lidar_range=lidar_ranges[i]
            )
        )
    return transformed_C_lidars

def point_cloud(C_lidars: list[np.array], lidar_ranges: np.array, lidar_h_angles: np.array, lidar_v_angles: np.array) -> list[np.array]:
    transformed_C_lidars = []
    for i in range(len(C_lidars)):
        transformed_C_lidars.append(distances_to_point_cloud(
            C_lidar=C_lidars[i],
            lidar_h_angle=lidar_h_angles[i],
            lidar_v_angle=lidar_v_angles[i],
            lidar_range=lidar_ranges[i]
            )
        )
    return transformed_C_lidars

def graph(C_lidars: list[np.array], lidar_ranges: np.array, lidar_h_angles: np.array, lidar_v_angles: np.array) -> list[np.array]:
    transformed_C_lidars = []
    for i in range(len(C_lidars)):
        transformed_C_lidars.append(distances_to_graph(
            C_lidar=C_lidars[i],
            lidar_h_angle=lidar_h_angles[i],
            lidar_v_angle=lidar_v_angles[i],
            lidar_range=lidar_ranges[i]
            )
        )
    return transformed_C_lidars

def read_csv(path: str) -> np.array:
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        return np.array(list(reader), dtype=np.float64)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Transformations to receive the representations mentioned in the SynthCave paper")
    parser.add_argument("--lidar_data", nargs='+', help="Array of csv file paths containing lidar data")
    parser.add_argument("--lidar_ranges", nargs='+', help="Array of lidar ranges in meters")
    parser.add_argument("--lidar_h_angles", nargs='+', help="Array of lidar horizontal angles in degrees")
    parser.add_argument("--lidar_v_angles", nargs='+', help="Array of lidar vertical angles in degrees")
    parser.add_argument("--rep", help="Representation to apply. One of 'depth_image', 'point_cloud', 'graph'")
    parser.add_argument("--output", help="folder path to write output data to")
    args = parser.parse_args()
    # Load data
    C_lidars = [read_csv(lidar) for lidar in args.lidars]
    lidar_ranges = np.array(args.lidar_ranges, dtype=np.float64)
    lidar_h_angles = np.array(args.lidar_h_angles, dtype=np.float64)
    lidar_v_angles = np.array(args.lidar_v_angles, dtype=np.float64)
    # Apply transformation
    if args.rep == "depth_image":
        transformed_C_lidars = depth_image(C_lidars, lidar_ranges, lidar_h_angles, lidar_v_angles)
    elif args.rep == "point_cloud":
        transformed_C_lidars = point_cloud(C_lidars, lidar_ranges, lidar_h_angles, lidar_v_angles)
    elif args.rep == "graph":
        transformed_C_lidars = graph(C_lidars, lidar_ranges, lidar_h_angles, lidar_v_angles)
    else:
        raise Exception("Unknown representation: " + args.rep)
    # Write data
    for i in range(len(transformed_C_lidars)):
        np.save(args.output + f"/lidar_{i}_{args.rep}.npy", transformed_C_lidars[i])

