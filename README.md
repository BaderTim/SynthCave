"# SynthCave" 

## Installation

### [PSTNet](https://github.com/hehefan/Point-Spatio-Temporal-Convolution)
Compile the CUDA layers for PointNet++, which are used for furthest point sampling (FPS) and radius neighbouring search:
```
cd src\model\PSTNet\modules\pointnet2_ops_lib
python setup.py install
```
To see if the compilation is successful, try to run `python src/model/PSTNet/modules/pst_convolutions.py` or `python src/model/PSTNet/PSTNet.py` to see if a forward pass works.