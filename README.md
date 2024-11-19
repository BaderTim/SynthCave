# SynthCave [![PyTest](https://github.com/BaderTim/SynthCave/actions/workflows/run_pytests.yml/badge.svg?branch=main)](https://github.com/BaderTim/SynthCave/actions/workflows/run_pytests.yml)

This repository contains the code for the dataset generation, as well as the code for the models used in the SynthCave paper. The models are implemented in PyTorch and include a **CNN**, **ASTGCN**, **TSViT**, and **PSTNet**. 

[View Paper](https://github.com/BaderTim/SynthCave/blob/main/Paper.pdf)


## About

**SynthCave** is a synthetic dataset for **3D odometry estimation** in cave-like environments. It contains synthetic LiDAR data in three different forms: **point clouds**, **depth-images**, and **graphs**, along with IMU and ground-truth data.   


The raw data was recorded with simulated LiDAR and IMU sensors, using the [Minecraft Measurement Mod](https://github.com/BaderTim/minecraft-measurement-mod) with Minecaft version 1.20.2. The dataset is publicly available on [Kaggle](https://www.kaggle.com/datasets/badertim/synthcave-3d-odometry-estimation).


## Installation

1) **Install requirements:**
    ```
    pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
    pip install -r requirements.txt
    ```

2) **Compile PointNet++ CUDA layers for
[PSTNet](https://github.com/hehefan/Point-Spatio-Temporal-Convolution):**
    ```
    cd baselines/model/PSTNet/modules/pointnet2_ops_lib
    python setup.py install
    ```
    PSTNet is the only model that requires CUDA and cannot be run on CPU.

3) **Test installation:**
    ```
    pytest baselines/model
    pytest synthcave
    ```
 
 4) **(Optional)** Manually test the forward pass of every model:
    - [CNN](https://research.engr.oregonstate.edu/rdml/sites/research.engr.oregonstate.edu.rdml/files/final_deep_learning_lidar_odometry.pdf): `python baselines/model/CNN/CNN.py` 
    - [ASTGCN](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#temporal-graph-attention-layers): `python baselines/model/ASTGCN/ASTGCN.py`
    - [TSViT](https://github.com/michaeltrs/DeepSatModels/tree/main?tab=readme-ov-file): `python baselines/model/TSViTcls/TSViT.py`
    - [PSTNet](https://github.com/hehefan/Point-Spatio-Temporal-Convolution): `python baselines/model/PSTNet/PSTNet.py`

