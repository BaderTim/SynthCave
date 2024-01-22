"""
derive_data.py

This script applies Structured Domain Randomization (SDR) as defined in the SynthCave paper to lidar and IMU data in the form of .npy files.

Required Arguments:
-------------------
--source: Folder path to read data from.
--target: Folder path to write output data to.
--devices: JSON file containing the device parameters.

Example Usage:
--------------
python synthcave/derive_data.py --source=data/2_cleaning --target=data/3_derived --devices=data/devices.json

The script reads lidar and IMU data from the specified files, applies SDR, and saves the SDR-processed data to the output folder.

Author:
-------
Tim Bader
Date: January 21, 2024
"""
import numpy as np
import argparse
import shutil
import json
import os

from sdr.modules.imu_sdr import cal, mag
from sdr.modules.lidar_sdr import ref, tf, vol


def sdr_1(C_lidar: np.array, lidar: dict) -> np.array:
    N, V, H = C_lidar.shape
    # reshape to (N, H, V)
    C_lidar = C_lidar.reshape(N, H, V)
    C_lidar = tf(C_lidar)
    occ_range = N if 100 > N else 100
    max_length = N if 50 > N else 50
    C_lidar = ref(C_lidar, max_range=lidar["range"], occurence_range=occ_range, max_length=max_length)
    C_lidar = vol(C_lidar, max_range=lidar["range"], occurence_range=occ_range, max_length=max_length)
    return C_lidar


def sdr_2(C_imu: np.array) -> np.array:
    N, _ = C_imu.shape
    occ_range = N if 100 > N else 100
    max_length = N if 50 > N else 50
    C_imu = cal(C_imu)
    C_imu = mag(C_imu, occurence_range=occ_range, max_length=max_length)
    return C_imu    


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Clean data from the original dataset and save it as numpy arrays")
    parser.add_argument("--source", help="Folder path to read data from")
    parser.add_argument("--target", help="Folder path to write output data to")
    parser.add_argument("--devices", help="JSON file containing the device parameters")
    args = parser.parse_args()

    devices = json.load(open(args.devices, "r"))
    lidar1 = devices["lidar1"]
    lidar2 = devices["lidar2"]
    lidar3 = devices["lidar3"]
    imu1 = devices["imu1"]

    # loop through subfolders in source folder
    for root, dirs, _ in os.walk(args.source):
        dirs.sort()
        for i, dir in enumerate(dirs):
            source_dir = os.path.join(root, dir)
            print(f"({i+1}/{len(dirs)}) Processing {dir}")
            # read data
            C_lidar1 = np.load(os.path.join(source_dir, "lidar1.npy"))
            C_lidar2 = np.load(os.path.join(source_dir, "lidar2.npy"))
            C_lidar3 = np.load(os.path.join(source_dir, "lidar3.npy"))
            C_imu1 = np.load(os.path.join(source_dir, "imu1.npy"))
            # apply SDR
            C_lidar1 = sdr_1(C_lidar1, lidar1)
            C_lidar2 = sdr_1(C_lidar2, lidar2)
            C_lidar3 = sdr_1(C_lidar3, lidar3)
            C_imu1 = sdr_2(C_imu1)
            # create target folder
            target_dir = source_dir.replace(args.source, args.target, 1)
            os.makedirs(target_dir, exist_ok=True)
            # save data
            np.save(os.path.join(target_dir, "lidar1.npy"), C_lidar1)
            np.save(os.path.join(target_dir, "lidar2.npy"), C_lidar2)
            np.save(os.path.join(target_dir, "lidar3.npy"), C_lidar3)
            np.save(os.path.join(target_dir, "imu1.npy"), C_imu1)
            # copy ground truths
            shutil.copy(os.path.join(source_dir, "lidar1_gt.npy"), target_dir)
            shutil.copy(os.path.join(source_dir, "lidar2_gt.npy"), target_dir)
            shutil.copy(os.path.join(source_dir, "lidar3_gt.npy"), target_dir)
            shutil.copy(os.path.join(source_dir, "imu1_gt.npy"), target_dir)

    print("Done")
