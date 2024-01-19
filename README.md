# SynthCave 

## Installation

1) **Install requirements:**
    ```
    pip install -r requirements.txt
    ```

2) **Compile PointNet++ CUDA layers for
[PSTNet](https://github.com/hehefan/Point-Spatio-Temporal-Convolution):**
    ```
    cd src/model/PSTNet/modules/pointnet2_ops_lib
    python setup.py install
    ```
    PSTNet is the only model that requires CUDA and cannot be run on CPU.

3) **Test installation:**
    ```
    pytest src/model
    ```
 
 4) **(Optional)** Manually test the forward pass of every model:
    - [ICP](https://users.soe.ucsc.edu/~davis/papers/Mapping_IROS04/IROS04diebel.pdf): `python src/model/ICP/ICP.py`
    - [CNN](https://research.engr.oregonstate.edu/rdml/sites/research.engr.oregonstate.edu.rdml/files/final_deep_learning_lidar_odometry.pdf): `python src/model/CNN/CNN.py` 
    - [ASTGCN](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#temporal-graph-attention-layers): `python src/model/ASTGCN/ASTGCN.py`
    - [TSViT](https://github.com/michaeltrs/DeepSatModels/tree/main?tab=readme-ov-file): `python src/model/TSViTcls/TSViT.py`
    - [PSTNet](https://github.com/hehefan/Point-Spatio-Temporal-Convolution): `python src/model/PSTNet/PSTNet.py`