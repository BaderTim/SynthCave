import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from model.ICP.module import icp

class ICP(nn.Module):
    def __init__(self, K=2):
        super(ICP, self).__init__()

        self.dataset_type = "point"
        self.K = K

    def forward(self, xyzs, imu_data=None):
        B, K, C, N = xyzs.shape
        res = torch.zeros((B, 5))
        # Perform ICP
        for b in range(B):
            prev_point_cloud = xyzs[b, 0, :, :].cpu().numpy()
            curr_point_cloud = xyzs[b, 1, :, :].cpu().numpy()
            T, distances, i = icp(prev_point_cloud, curr_point_cloud, max_iterations=200)
            # T is (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
            # Decompose the transformation matrix T
            R = T[:3, :3]  # rotation matrix
            t = T[:3, 3]   # translation vector
            # Convert the rotation matrix to Euler angles
            r = Rot.from_matrix(R)
            thetaX, thetaY, thetaZ = r.as_euler('xyz')
            # Switch y and z
            thetaY, thetaZ = thetaZ, thetaY
            # Convert Euler angles (roll, pitch, yaw) to spherical coordinates (theta, phi)
            theta = np.arctan2(np.sqrt(thetaX**2 + thetaY**2), thetaZ)
            phi = np.arctan2(thetaY, thetaX)
            # Extract x, y, z changes
            x, z, y = t[0], t[1], t[2]
            # Store the results
            res[b, :] = torch.tensor([x, y, z, theta, phi])
 
        return res.to(xyzs.device)


if __name__ == "__main__":

    K = 2
    model = ICP(K=K)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    x = torch.randn(8, K, 50_200, 3) # (B, L, C, N)

    out = model(x)

    print("Shape of out :", out.shape) 