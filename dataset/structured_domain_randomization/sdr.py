import numpy as np
import argparse
import csv

from modules.imu_sdr import cal, mag
from modules.lidar_sdr import ref, tf, vol


def sdr_1(C_lidars: list[np.array]) -> list[np.array]:
    res = []
    for C_lidar in C_lidars:
        res.append(vol(ref(tf(C_lidar))))
    return res


def sdr_2(C_imu: np.array) -> np.array:
    return mag(cal(C_imu))


def sdr(C_lidars: list[np.array], C_imu: np.array) -> (list[np.array], np.array):
    return sdr_1(C_lidars), [sdr_2(C_imu)]


def read_csv(path: str) -> np.array:
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        return np.array(list(reader), dtype=np.float64)
    

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Structured Domain Randomization for SynthCave")
    parser.add_argument("--lidars", nargs='+', help="Array of csv file paths containing lidar data")
    parser.add_argument("--imu", help="csv file path containing imu data")
    parser.add_argument("--output", help="folder path to write output data to")
    args = parser.parse_args()
    # Load data
    C_lidars = [read_csv(lidar) for lidar in args.lidars]
    C_imu = read_csv(args.imu)
    # Apply SDR
    C_lidars_sdr, C_imu_sdr = sdr(C_lidars, C_imu)
    # Write data
    for i in range(len(C_lidars_sdr)):
        np.savetxt(args.output + "/lidar_" + str(i) + ".csv", C_lidars_sdr[i], delimiter=";")
    np.savetxt(args.output + "/imu.csv", C_imu_sdr, delimiter=";")
