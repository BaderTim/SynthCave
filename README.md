# SynthCave [![PyTest](https://github.com/BaderTim/SynthCave/actions/workflows/run_pytests.yml/badge.svg?branch=main)](https://github.com/BaderTim/SynthCave/actions/workflows/run_pytests.yml)

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
    ```
 
 4) **(Optional)** Manually test the forward pass of every model:
    - [ICP](https://users.soe.ucsc.edu/~davis/papers/Mapping_IROS04/IROS04diebel.pdf): `python baselines/model/ICP/ICP.py`
    - [CNN](https://research.engr.oregonstate.edu/rdml/sites/research.engr.oregonstate.edu.rdml/files/final_deep_learning_lidar_odometry.pdf): `python baselines/model/CNN/CNN.py` 
    - [ASTGCN](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#temporal-graph-attention-layers): `python baselines/model/ASTGCN/ASTGCN.py`
    - [TSViT](https://github.com/michaeltrs/DeepSatModels/tree/main?tab=readme-ov-file): `python baselines/model/TSViTcls/TSViT.py`
    - [PSTNet](https://github.com/hehefan/Point-Spatio-Temporal-Convolution): `python baselines/model/PSTNet/PSTNet.py`