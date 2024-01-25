"""
calculate_statistics.py

Calculate statistics of the dataset.

Required Arguments:
-------------------
--source: Folder path to read data from.

Example Usage:
--------------
python synthcave/calculate_statistics.py --source=data/2_cleaning

Author:
-------
Tim Bader
Date: January 21, 2024
"""
import numpy as np
import argparse
import json
import os


def get_stats_for_section(C_lidar_gt: np.array, hz=5) -> dict:
    """
    Calculate the statistics for a section of the dataset.

    Arguments:
    ----------
    C_lidar_gt: Numpy array containing the LiDAR GT data in the shape N x 6.
                like [['posX', 'posY', 'posZ', 'rotX', 'rotY', 'rotZ', 'theta', 'phi'], ...]

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
    stats["theta_total"] = np.sum(np.abs(C_lidar_gt[:, 6]))
    stats["phi_total"] = np.sum(np.abs(C_lidar_gt[:, 7]))
    stats["theta_rad_rounded"] = [round(np.deg2rad(theta), 1) for theta in C_lidar_gt[:, 6]]
    stats["phi_rad_rounded"] = [round(np.deg2rad(phi), 1) for phi in C_lidar_gt[:, 7]]
    stats["x_rounded"] = [round(x, 1) for x in C_lidar_gt[:, 0]]
    stats["y_rounded"] = [round(y, 1) for y in C_lidar_gt[:, 1]]
    stats["z_rounded"] = [round(z, 1) for z in C_lidar_gt[:, 2]]
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
                abs_stats[section]["phi_rad_rounded"] = stats["phi_rad_rounded"]
                abs_stats[section]["theta_rad_rounded"] = stats["theta_rad_rounded"]
                abs_stats[section]["x_rounded"] = stats["x_rounded"]
                abs_stats[section]["y_rounded"] = stats["y_rounded"]
                abs_stats[section]["z_rounded"] = stats["z_rounded"]
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
                abs_stats[section]["phi_rad_rounded"] += stats["phi_rad_rounded"]
                abs_stats[section]["theta_rad_rounded"] += stats["theta_rad_rounded"]
                abs_stats[section]["x_rounded"] += stats["x_rounded"]
                abs_stats[section]["y_rounded"] += stats["y_rounded"]
                abs_stats[section]["z_rounded"] += stats["z_rounded"]


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

    theta_rad_rounded = np.concatenate([abs_stats[key]["theta_rad_rounded"] for key in abs_stats])
    phi_rad_rounded = np.concatenate([abs_stats[key]["phi_rad_rounded"] for key in abs_stats])
    x_rounded = np.concatenate([abs_stats[key]["x_rounded"] for key in abs_stats])
    y_rounded = np.concatenate([abs_stats[key]["y_rounded"] for key in abs_stats])
    z_rounded = np.concatenate([abs_stats[key]["z_rounded"] for key in abs_stats])
    theta_rad_rounded_hist = np.unique(theta_rad_rounded, return_counts=True)
    phi_rad_rounded_hist = np.unique(phi_rad_rounded, return_counts=True)
    x_rounded_hist = np.unique(x_rounded, return_counts=True)
    y_rounded_hist = np.unique(y_rounded, return_counts=True)
    z_rounded_hist = np.unique(z_rounded, return_counts=True)
    upper_limit = 1
    lower_limit = -1
    theta_rad_rounded_hist = (theta_rad_rounded_hist[0][np.where((theta_rad_rounded_hist[0] >= lower_limit) & (theta_rad_rounded_hist[0] <= upper_limit))], theta_rad_rounded_hist[1][np.where((theta_rad_rounded_hist[0] >= lower_limit) & (theta_rad_rounded_hist[0] <= upper_limit))])
    phi_rad_rounded_hist = (phi_rad_rounded_hist[0][np.where((phi_rad_rounded_hist[0] >= lower_limit) & (phi_rad_rounded_hist[0] <= upper_limit))], phi_rad_rounded_hist[1][np.where((phi_rad_rounded_hist[0] >= lower_limit) & (phi_rad_rounded_hist[0] <= upper_limit))])
    x_rounded_hist = (x_rounded_hist[0][np.where((x_rounded_hist[0] >= lower_limit) & (x_rounded_hist[0] <= upper_limit))], x_rounded_hist[1][np.where((x_rounded_hist[0] >= lower_limit) & (x_rounded_hist[0] <= upper_limit))])
    y_rounded_hist = (y_rounded_hist[0][np.where((y_rounded_hist[0] >= lower_limit) & (y_rounded_hist[0] <= upper_limit))], y_rounded_hist[1][np.where((y_rounded_hist[0] >= lower_limit) & (y_rounded_hist[0] <= upper_limit))])
    z_rounded_hist = (z_rounded_hist[0][np.where((z_rounded_hist[0] >= lower_limit) & (z_rounded_hist[0] <= upper_limit))], z_rounded_hist[1][np.where((z_rounded_hist[0] >= lower_limit) & (z_rounded_hist[0] <= upper_limit))])
    tot_stats["theta_rad_rounded_hist"] = (np.round(theta_rad_rounded_hist[0], 1), theta_rad_rounded_hist[1])
    tot_stats["theta_rad_rounded_hist_out_of_range"] = np.abs(len(theta_rad_rounded) - np.sum(theta_rad_rounded_hist[1]))
    tot_stats["phi_rad_rounded_hist"] = (np.round(phi_rad_rounded_hist[0], 1), phi_rad_rounded_hist[1])
    tot_stats["phi_rad_rounded_hist_out_of_range"] = np.abs(len(phi_rad_rounded) - np.sum(phi_rad_rounded_hist[1]))
    tot_stats["x_rounded_hist"] = (np.round(x_rounded_hist[0], 1), x_rounded_hist[1])
    tot_stats["x_rounded_hist_out_of_range"] = np.abs(len(x_rounded) - np.sum(x_rounded_hist[1]))
    tot_stats["y_rounded_hist"] = (np.round(y_rounded_hist[0], 1), y_rounded_hist[1])
    tot_stats["y_rounded_hist_out_of_range"] = np.abs(len(y_rounded) - np.sum(y_rounded_hist[1]))
    tot_stats["z_rounded_hist"] = (np.round(z_rounded_hist[0], 1), z_rounded_hist[1])
    tot_stats["z_rounded_hist_out_of_range"] = np.abs(len(z_rounded) - np.sum(z_rounded_hist[1]))

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
        del abs_stats[key]["phi_rad_rounded"]
        del abs_stats[key]["theta_rad_rounded"]
        del abs_stats[key]["x_rounded"]
        del abs_stats[key]["y_rounded"]
        del abs_stats[key]["z_rounded"]

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
