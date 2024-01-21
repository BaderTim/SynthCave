"""
calculate_statistics.py

Calculate statistics of the dataset.

Required Arguments:
-------------------
--source: Folder path to read data from.

Example Usage:
--------------
python dataset/calculate_statistics.py --source=data/2_cleaning

Author:
-------
Tim Bader
Date: January 21, 2024
"""
import numpy as np
import argparse
import json
import math
import os


def to_spherical(x, y, z):
    """Converts a cartesian coordinate (x, y, z) into a spherical one (theta, phi)."""
    theta = math.atan2(math.sqrt(x * x + y * y), z)
    phi = math.atan2(y, x)
    return (theta, phi)


def get_stats_for_section(C_lidar_gt: np.array, hz=5) -> dict:
    """
    Calculate the statistics for a section of the dataset.

    Arguments:
    ----------
    C_lidar_gt: Numpy array containing the LiDAR GT data in the shape N x 6.
                like [['posX', 'posY', 'posZ', 'rotX', 'rotY', 'rotZ'], ...]

    Returns:
    --------
    stats: Dictionary containing the statistics for the section.
    """
    stats = {}
    stats["frames"] = len(C_lidar_gt)
    stats["duration"] = len(C_lidar_gt) / hz
    stats["x_distance_total"] = np.sum(np.abs(C_lidar_gt[:, 0]))
    stats["y_distance_total"] = np.sum(np.abs(C_lidar_gt[:, 1]))
    stats["z_distance_total"] = np.sum(np.abs(C_lidar_gt[:, 2]))
    stats["xz_distance_total"] = np.sum(np.sqrt(np.abs(C_lidar_gt[:, 0])**2 + np.abs(C_lidar_gt[:, 2])**2))
    stats["x_rotation_total"] = np.sum(np.abs(C_lidar_gt[:, 3]))
    stats["y_rotation_total"] = np.sum(np.abs(C_lidar_gt[:, 4]))
    stats["z_rotation_total"] = np.sum(np.abs(C_lidar_gt[:, 5]))
    phi_total = 0
    theta_total = 0
    for i in range(len(C_lidar_gt)):
        theta, phi = to_spherical(x=C_lidar_gt[i, 3], y=C_lidar_gt[i, 5], z=C_lidar_gt[i, 4])
        theta_total += abs(theta)
        phi_total += abs(phi)
    stats["phi_total"] = phi_total
    stats["theta_total"] = theta_total
    return stats


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Clean data from the original dataset and save it as numpy arrays")
    parser.add_argument("--source", help="Folder path to read data from")
    args = parser.parse_args()

    abs_stats = {}
    
    # loop through subfolders in source folder
    for root, dirs, _ in os.walk(args.source):

        dirs.sort()

        for i, dir in enumerate(dirs):
            source_dir = os.path.join(root, dir)
            print(f"({i+1}/{len(dirs)}) Processing {dir}")
            section, number = dir.rsplit('_', 1)

            C_lidar1 = np.load(os.path.join(source_dir, "lidar1_gt.npy"))
            stats = get_stats_for_section(C_lidar1)
            if section not in abs_stats: # init lists
                abs_stats[section] = {}
                abs_stats[section]["frames_list"] = [stats["frames"]]
                abs_stats[section]["duration_list"] = [stats["duration"]]
                abs_stats[section]["x_distance_list"] = [stats["x_distance_total"]]
                abs_stats[section]["y_distance_list"] = [stats["y_distance_total"]]
                abs_stats[section]["z_distance_list"] = [stats["z_distance_total"]]
                abs_stats[section]["xz_distance_list"] = [stats["xz_distance_total"]]
                abs_stats[section]["x_rotation_list"] = [stats["x_rotation_total"]]
                abs_stats[section]["y_rotation_list"] = [stats["y_rotation_total"]]
                abs_stats[section]["z_rotation_list"] = [stats["z_rotation_total"]]
                abs_stats[section]["phi_list"] = [stats["phi_total"]]
                abs_stats[section]["theta_list"] = [stats["theta_total"]]
            else: # add on top as array
                abs_stats[section]["frames_list"].append(stats["frames"])
                abs_stats[section]["duration_list"].append(stats["duration"])
                abs_stats[section]["x_distance_list"].append(stats["x_distance_total"])
                abs_stats[section]["y_distance_list"].append(stats["y_distance_total"])
                abs_stats[section]["z_distance_list"].append(stats["z_distance_total"])
                abs_stats[section]["xz_distance_list"].append(stats["xz_distance_total"])
                abs_stats[section]["x_rotation_list"].append(stats["x_rotation_total"])
                abs_stats[section]["y_rotation_list"].append(stats["y_rotation_total"])
                abs_stats[section]["z_rotation_list"].append(stats["z_rotation_total"])
                abs_stats[section]["phi_list"].append(stats["phi_total"])
                abs_stats[section]["theta_list"].append(stats["theta_total"])


    # calculate total and median values for all sections
    tot_stats = {}
    tot_stats["frames_total"] = np.sum([abs_stats[key]["frames_list"] for key in abs_stats])
    tot_stats["duration_total"] = np.sum([abs_stats[key]["duration_list"] for key in abs_stats])
    tot_stats["x_distance_total"] = np.sum([abs_stats[key]["x_distance_list"] for key in abs_stats])
    tot_stats["y_distance_total"] = np.sum([abs_stats[key]["y_distance_list"] for key in abs_stats])
    tot_stats["z_distance_total"] = np.sum([abs_stats[key]["z_distance_list"] for key in abs_stats])
    tot_stats["xz_distance_total"] = np.sum([abs_stats[key]["xz_distance_list"] for key in abs_stats])
    tot_stats["x_rotation_total"] = np.sum([abs_stats[key]["x_rotation_list"] for key in abs_stats])
    tot_stats["y_rotation_total"] =  np.sum([abs_stats[key]["y_rotation_list"] for key in abs_stats])
    tot_stats["z_rotation_total"] = np.sum([abs_stats[key]["z_rotation_list"] for key in abs_stats])
    tot_stats["phi_total"] = np.sum([abs_stats[key]["phi_list"] for key in abs_stats])
    tot_stats["theta_total"] = np.sum([abs_stats[key]["theta_list"] for key in abs_stats])

    tot_stats["frames_median"] = np.median(np.concatenate([abs_stats[key]["frames_list"] for key in abs_stats]))
    tot_stats["duration_median"] = np.median(np.concatenate([abs_stats[key]["duration_list"] for key in abs_stats]))
    tot_stats["x_distance_median"] = np.median(np.concatenate([abs_stats[key]["x_distance_list"] for key in abs_stats]))
    tot_stats["y_distance_median"] = np.median(np.concatenate([abs_stats[key]["y_distance_list"] for key in abs_stats]))
    tot_stats["z_distance_median"] = np.median(np.concatenate([abs_stats[key]["z_distance_list"] for key in abs_stats]))
    tot_stats["xz_distance_median"] = np.median(np.concatenate([abs_stats[key]["xz_distance_list"] for key in abs_stats]))
    tot_stats["x_rotation_median"] = np.median(np.concatenate([abs_stats[key]["x_rotation_list"] for key in abs_stats]))
    tot_stats["y_rotation_median"] = np.median(np.concatenate([abs_stats[key]["y_rotation_list"] for key in abs_stats]))
    tot_stats["z_rotation_median"] = np.median(np.concatenate([abs_stats[key]["z_rotation_list"] for key in abs_stats]))
    tot_stats["phi_median"] = np.median(np.concatenate([abs_stats[key]["phi_list"] for key in abs_stats]))
    tot_stats["theta_median"] = np.median(np.concatenate([abs_stats[key]["theta_list"] for key in abs_stats]))

    tot_stats["x_rotation_per_second"] = tot_stats["x_rotation_total"] / tot_stats["duration_total"]
    tot_stats["y_rotation_per_second"] = tot_stats["y_rotation_total"] / tot_stats["duration_total"]
    tot_stats["z_rotation_per_second"] = tot_stats["z_rotation_total"] / tot_stats["duration_total"]
    tot_stats["phi_per_second"] = tot_stats["phi_total"] / tot_stats["duration_total"]
    tot_stats["theta_per_second"] = tot_stats["theta_total"] / tot_stats["duration_total"]

    # calculate additional total and median values per section
    for key in abs_stats:
        abs_stats[key]["frames_total"] = np.sum(abs_stats[key]["frames_list"])
        abs_stats[key]["duration_total"] = np.sum(abs_stats[key]["duration_list"])
        abs_stats[key]["x_distance_total"] = np.sum(abs_stats[key]["x_distance_list"])
        abs_stats[key]["y_distance_total"] = np.sum(abs_stats[key]["y_distance_list"])
        abs_stats[key]["z_distance_total"] = np.sum(abs_stats[key]["z_distance_list"])
        abs_stats[key]["xz_distance_total"] = np.sum(abs_stats[key]["xz_distance_list"])
        abs_stats[key]["x_rotation_total"] = np.sum(abs_stats[key]["x_rotation_list"])
        abs_stats[key]["y_rotation_total"] = np.sum(abs_stats[key]["y_rotation_list"])
        abs_stats[key]["z_rotation_total"] = np.sum(abs_stats[key]["z_rotation_list"])
        abs_stats[key]["phi_total"] = np.sum(abs_stats[key]["phi_list"])
        abs_stats[key]["theta_total"] = np.sum(abs_stats[key]["theta_list"])

        abs_stats[key]["frames_median"] = np.median(abs_stats[key]["frames_list"])
        abs_stats[key]["duration_median"] = np.median(abs_stats[key]["duration_list"])
        abs_stats[key]["x_distance_median"] = np.median(abs_stats[key]["x_distance_list"])
        abs_stats[key]["y_distance_median"] = np.median(abs_stats[key]["y_distance_list"])
        abs_stats[key]["z_distance_median"] = np.median(abs_stats[key]["z_distance_list"])
        abs_stats[key]["xz_distance_median"] = np.median(abs_stats[key]["xz_distance_list"])
        abs_stats[key]["x_rotation_median"] = np.median(abs_stats[key]["x_rotation_list"])
        abs_stats[key]["y_rotation_median"] = np.median(abs_stats[key]["y_rotation_list"])
        abs_stats[key]["z_rotation_median"] = np.median(abs_stats[key]["z_rotation_list"])
        abs_stats[key]["phi_median"] = np.median(abs_stats[key]["phi_list"])
        abs_stats[key]["theta_median"] = np.median(abs_stats[key]["theta_list"])
        
        abs_stats[key]["x_rotation_per_second"] = abs_stats[key]["x_rotation_total"] / abs_stats[key]["duration_total"]
        abs_stats[key]["y_rotation_per_second"] = abs_stats[key]["y_rotation_total"] / abs_stats[key]["duration_total"]
        abs_stats[key]["z_rotation_per_second"] = abs_stats[key]["z_rotation_total"] / abs_stats[key]["duration_total"]
        abs_stats[key]["phi_per_second"] = abs_stats[key]["phi_total"] / abs_stats[key]["duration_total"]
        abs_stats[key]["theta_per_second"] = abs_stats[key]["theta_total"] / abs_stats[key]["duration_total"]

        # remove lists
        del abs_stats[key]["frames_list"]
        del abs_stats[key]["duration_list"]
        del abs_stats[key]["x_distance_list"]
        del abs_stats[key]["y_distance_list"]
        del abs_stats[key]["z_distance_list"]
        del abs_stats[key]["xz_distance_list"]
        del abs_stats[key]["x_rotation_list"]
        del abs_stats[key]["y_rotation_list"]
        del abs_stats[key]["z_rotation_list"]
        del abs_stats[key]["phi_list"]
        del abs_stats[key]["theta_list"]

    abs_stats["total"] = tot_stats


    def convert(o):
        if isinstance(o, np.int64): return int(o)  
        if isinstance(o, np.float64): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, np.int32): return int(o)
        if isinstance(o, np.float32): return float(o)
        raise TypeError

    # save as json
    with open(os.path.join(args.source, "lidar1_stats.json"), "w") as f:
        json.dump(abs_stats, f, default=convert)

    print("Done")
