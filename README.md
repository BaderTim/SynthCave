"# SynthCave" 

## Installation

### [PSTNet](https://github.com/hehefan/Point-Spatio-Temporal-Convolution)
Compile the CUDA layers for PointNet++, which are used for furthest point sampling (FPS) and radius neighbouring search:
```
cd src\model\PSTNet\modules\pointnet2_ops_lib
python setup.py install
```
To see if the compilation is successful, try to run `python src/model/PSTNet/modules/pst_convolutions.py` or `python src/model/PSTNet/PSTNet.py` to see if a forward pass works.

### [CNN](https://research.engr.oregonstate.edu/rdml/sites/research.engr.oregonstate.edu.rdml/files/final_deep_learning_lidar_odometry.pdf)
No manual installation required. Try to run `python src/model/CNN/CNN.py` to see if a forward pass works.

### [ASTGCN](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#temporal-graph-attention-layers)
No manual installation required. Try to run `python src/model/ASTGCN/ASTGCN.py` to see if a forward pass works.