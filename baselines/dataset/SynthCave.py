import os
import torch
import numpy as np
from torch.utils.data import Dataset

class GraphDataset(Dataset):
    def __init__(self, data_folder: str, frames: int, gt_as_rad: bool = True, gt_limit: None | list = [-1, 1]):
        """
        Args:
            data_folder (string): Path to the graph dataset's train/val folder.
                                  In each subfolder, there should be a labels.csv file and a folder for each sample. 
            frames (int): Number of frames in each sample.
            gt_as_rad (bool): Whether to return the ground truth as radians or not.
            gt_limit (None | list): If not None, the ground truth will be limited to the given range.
        """
        self.path = data_folder
        self.frames = frames
        self.gt_as_rad = gt_as_rad
        self.gt_limit = gt_limit
        self.theta_rounded_hist = []
        self.phi_rounded_hist = []
        self.x_rounded_hist = []
        self.y_rounded_hist = []
        self.z_rounded_hist = []
        # the keys represent the cumulative number of samples
        self.index = {}
        self.samples = 0
        print(f"Initializing dataset from '{self.path}'...")
        # loop through folders
        for file in os.listdir(self.path):
            filename = os.fsdecode(file)
            if filename.endswith("_gt.npy"): # load sequence set at once and not after another
                sequence_id = int(filename.split("_")[0])
                graphs = np.load(os.path.join(self.path, f"{sequence_id}_graph.npy"))
                imus = np.load(os.path.join(self.path, f"{sequence_id}_imu.npy"))
                gts = np.load(os.path.join(self.path, f"{sequence_id}_gt.npy"))
                if graphs.shape[0] != imus.shape[0] or graphs.shape[0] != gts.shape[0]:
                    raise Exception(f"In sequence {sequence_id}, the number of graphs ({graphs.shape[0]}), IMU ({imus.shape[0]}) and GT ({gts.shape[0]}) do not match.")
                samples_in_graphs = graphs.shape[0] - self.frames + 1
                self.index[self.samples] = sequence_id
                self.samples += samples_in_graphs
                # collect histogram data
                temp_x_rounded_hist = np.round(gts[:, 0], 1)
                temp_y_rounded_hist = np.round(gts[:, 1], 1)
                temp_z_rounded_hist = np.round(gts[:, 2], 1)
                temp_theta_rounded_hist = np.round(np.deg2rad(gts[:, 6]) if self.gt_as_rad else gts[:, 6], 1)
                temp_phi_rounded_hist = np.round(np.deg2rad(gts[:, 7]) if self.gt_as_rad else gts[:, 7], 1)
                if self.gt_limit is not None:
                    temp_x_rounded_hist = np.clip(temp_x_rounded_hist, self.gt_limit[0], self.gt_limit[1])
                    temp_y_rounded_hist = np.clip(temp_y_rounded_hist, self.gt_limit[0], self.gt_limit[1])
                    temp_z_rounded_hist = np.clip(temp_z_rounded_hist, self.gt_limit[0], self.gt_limit[1])
                    temp_theta_rounded_hist = np.clip(temp_theta_rounded_hist, self.gt_limit[0], self.gt_limit[1])
                    temp_phi_rounded_hist = np.clip(temp_phi_rounded_hist, self.gt_limit[0], self.gt_limit[1])
                self.x_rounded_hist += temp_x_rounded_hist.tolist()
                self.y_rounded_hist += temp_y_rounded_hist.tolist()
                self.z_rounded_hist += temp_z_rounded_hist.tolist()
                self.theta_rounded_hist += temp_theta_rounded_hist.tolist()
                self.phi_rounded_hist += temp_phi_rounded_hist.tolist()
                del graphs, imus, gts
        # create histograms
        self.x_rounded_hist = np.unique(self.x_rounded_hist, return_counts=True)
        self.y_rounded_hist = np.unique(self.y_rounded_hist, return_counts=True)
        self.z_rounded_hist = np.unique(self.z_rounded_hist, return_counts=True)
        self.theta_rounded_hist = np.unique(self.theta_rounded_hist, return_counts=True)
        self.phi_rounded_hist = np.unique(self.phi_rounded_hist, return_counts=True)
        print(f"Dataset initialized with {self.samples} samples.")
    

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        """
        Returns:
            graph (torch.Tensor): The graph data of the sample, shape (N, F_in, T).
            edges (torch.Tensor): The edges data of the sample, shape (E, 2).
            imu (torch.Tensor): The IMU data of the sample, shape (6, T).
            gt (torch.Tensor): The ground truth data of the sample, shape (5,).
        """
        # find the sequence that contains the index
        keys = sorted(self.index.keys())
        for i, key in enumerate(keys):
            if idx < key:
                # means that the index is in previous key sequence
                full_graph = np.load(os.path.join(self.path, f"{self.index[keys[i-1]]}_graph.npy"))
                full_edges = np.load(os.path.join(self.path, f"{self.index[keys[i-1]]}_edges.npy"))
                full_imu = np.load(os.path.join(self.path, f"{self.index[keys[i-1]]}_imu.npy"))
                full_gt = np.load(os.path.join(self.path, f"{self.index[keys[i-1]]}_gt.npy"))
                # get the index of the sample in the sequence
                idx_in_sequence = idx - keys[i-1]
                # return the sample
                graph = torch.from_numpy(full_graph[idx_in_sequence:idx_in_sequence+self.frames]) # (T, N, F_in)
                edges = torch.from_numpy(full_edges[idx_in_sequence:idx_in_sequence+self.frames]) # (T, E, 2)
                imu_np = full_imu[idx_in_sequence:idx_in_sequence+self.frames] # (T, 6)
                imu = torch.from_numpy(imu_np) # (T, 6)
                gt_np = full_gt[idx_in_sequence+self.frames-1] # (8,)
                if self.gt_as_rad:
                    gt_np[6:8] = np.deg2rad(gt_np[6:8])
                if self.gt_limit is not None:
                    gt_np = np.clip(gt_np, self.gt_limit[0], self.gt_limit[1])
                gt = torch.from_numpy(gt_np)
                # remove variables from memory
                del full_graph, full_edges, full_imu, full_gt
                graph = graph.permute(1, 2, 0)
                edges = edges[0].permute(1, 0)
                gt = torch.tensor([gt[0], gt[1], gt[2], gt[6], gt[7]])
                return graph, edges, imu, gt
        # means that the index is in the last key sequence
        full_graph = np.load(os.path.join(self.path, f"{self.index[keys[-1]]}_graph.npy"))
        full_edges = np.load(os.path.join(self.path, f"{self.index[keys[-1]]}_edges.npy"))
        full_imu = np.load(os.path.join(self.path, f"{self.index[keys[-1]]}_imu.npy"))
        full_gt = np.load(os.path.join(self.path, f"{self.index[keys[-1]]}_gt.npy"))
        # get the index of the sample in the sequence
        idx_in_sequence = idx - keys[-1]
        # return the sample
        graph = torch.from_numpy(full_graph[idx_in_sequence:idx_in_sequence+self.frames]) # (T, N, F_in)
        edges = torch.from_numpy(full_edges[idx_in_sequence:idx_in_sequence+self.frames]) # (T, E, 2)
        imu_np = full_imu[idx_in_sequence:idx_in_sequence+self.frames] # (T, 6)
        gt_np = full_gt[idx_in_sequence+self.frames-1] # (8,)
        imu = torch.from_numpy(imu_np) # (T, 6)
        gt_np = full_gt[idx_in_sequence+self.frames-1] # (8,)
        if self.gt_as_rad:
            gt_np[6:8] = np.deg2rad(gt_np[6:8])
        if self.gt_limit is not None:
            gt_np = np.clip(gt_np, self.gt_limit[0], self.gt_limit[1])
        gt = torch.from_numpy(gt_np)
        del full_graph, full_edges, full_imu, full_gt
        graph = graph.permute(1, 2, 0)
        edges = edges[0].permute(1, 0)
        gt = torch.tensor([gt[0], gt[1], gt[2], gt[6], gt[7]])
        return graph, edges, imu, gt
    
    def gt_as_degree(self, gt: torch.Tensor):
        """
        Transforms the ground truth back to its original form.
        """
        return torch.tensor([gt[0], gt[1], gt[2], np.rad2deg(gt[3]), np.rad2deg(gt[4])])
            

class PointDataset(Dataset):
    def __init__(self, data_folder: str, frames: int, gt_as_rad: bool = True, gt_limit: None | list = [-1, 1]):
        """
        Args:
            data_folder (string): Path to the point cloud dataset's train/val folder.
                                  In each subfolder, there should be a labels.csv file and a folder for each sample. 
            frames (int): Number of frames in each sample.
            gt_as_rad (bool): Whether to return the ground truth as radians or not.
            gt_limit (None | list): If not None, the ground truth will be limited to the given range.
        """
        self.path = data_folder
        self.frames = frames
        self.gt_as_rad = gt_as_rad
        self.gt_limit = gt_limit
        self.theta_rounded_hist = []
        self.phi_rounded_hist = []
        self.x_rounded_hist = []
        self.y_rounded_hist = []
        self.z_rounded_hist = []
        # the keys represent the cumulative number of samples
        self.index = {}
        self.samples = 0
        print(f"Initializing dataset from '{self.path}'...")
        # loop through folders
        for file in os.listdir(self.path):
            filename = os.fsdecode(file)
            if filename.endswith("_gt.npy"): # load sequence set at once and not after another
                sequence_id = int(filename.split("_")[0])
                pcs = np.load(os.path.join(self.path, f"{sequence_id}_pc.npy"))
                imus = np.load(os.path.join(self.path, f"{sequence_id}_imu.npy"))
                gts = np.load(os.path.join(self.path, f"{sequence_id}_gt.npy"))
                if pcs.shape[0] != imus.shape[0] or pcs.shape[0] != gts.shape[0]:
                    raise Exception(f"In sequence {sequence_id}, the number of point clouds ({pcs.shape[0]}), IMU ({imus.shape[0]}) and GT ({gts.shape[0]}) do not match.")
                samples_in_pcs = pcs.shape[0] - self.frames + 1
                self.index[self.samples] = sequence_id
                self.samples += samples_in_pcs
                # collect histogram data
                temp_x_rounded_hist = np.round(gts[:, 0], 1)
                temp_y_rounded_hist = np.round(gts[:, 1], 1)
                temp_z_rounded_hist = np.round(gts[:, 2], 1)
                temp_theta_rounded_hist = np.round(np.deg2rad(gts[:, 6]) if self.gt_as_rad else gts[:, 6], 1)
                temp_phi_rounded_hist = np.round(np.deg2rad(gts[:, 7]) if self.gt_as_rad else gts[:, 7], 1)
                if self.gt_limit is not None:
                    temp_x_rounded_hist = np.clip(temp_x_rounded_hist, self.gt_limit[0], self.gt_limit[1])
                    temp_y_rounded_hist = np.clip(temp_y_rounded_hist, self.gt_limit[0], self.gt_limit[1])
                    temp_z_rounded_hist = np.clip(temp_z_rounded_hist, self.gt_limit[0], self.gt_limit[1])
                    temp_theta_rounded_hist = np.clip(temp_theta_rounded_hist, self.gt_limit[0], self.gt_limit[1])
                    temp_phi_rounded_hist = np.clip(temp_phi_rounded_hist, self.gt_limit[0], self.gt_limit[1])
                self.x_rounded_hist += temp_x_rounded_hist.tolist()
                self.y_rounded_hist += temp_y_rounded_hist.tolist()
                self.z_rounded_hist += temp_z_rounded_hist.tolist()
                self.theta_rounded_hist += temp_theta_rounded_hist.tolist()
                self.phi_rounded_hist += temp_phi_rounded_hist.tolist()
                del pcs, imus, gts
        # create histograms
        self.x_rounded_hist = np.unique(self.x_rounded_hist, return_counts=True)
        self.y_rounded_hist = np.unique(self.y_rounded_hist, return_counts=True)
        self.z_rounded_hist = np.unique(self.z_rounded_hist, return_counts=True)
        self.theta_rounded_hist = np.unique(self.theta_rounded_hist, return_counts=True)
        self.phi_rounded_hist = np.unique(self.phi_rounded_hist, return_counts=True)
        print(f"Dataset initialized with {self.samples} samples.")
    

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        """
        Returns:
            pc (torch.Tensor): The point cloud data of the sample, shape (T, P, 3).
            imu (torch.Tensor): The IMU data of the sample, shape (6, T).
            gt (torch.Tensor): The ground truth data of the sample, shape (5,).
        """
        # find the sequence that contains the index
        keys = sorted(self.index.keys())
        for i, key in enumerate(keys):
            if idx < key:
                # means that the index is in previous key sequence
                full_pc = np.load(os.path.join(self.path, f"{self.index[keys[i-1]]}_pc.npy"))
                full_imu = np.load(os.path.join(self.path, f"{self.index[keys[i-1]]}_imu.npy"))
                full_gt = np.load(os.path.join(self.path, f"{self.index[keys[i-1]]}_gt.npy"))
                # get the index of the sample in the sequence
                idx_in_sequence = idx - keys[i-1]
                # return the sample
                pc = torch.from_numpy(full_pc[idx_in_sequence:idx_in_sequence+self.frames]) # (T, P, 3)
                imu = torch.from_numpy(full_imu[idx_in_sequence:idx_in_sequence+self.frames]) # (T, 6)
                gt_np = full_gt[idx_in_sequence+self.frames-1] # (8,)
                if self.gt_as_rad:
                    gt_np[6:8] = np.deg2rad(gt_np[6:8])
                if self.gt_limit is not None:
                    gt_np = np.clip(gt_np, self.gt_limit[0], self.gt_limit[1])
                gt = torch.from_numpy(gt_np)
                # remove variables from memory
                del full_pc, full_imu, full_gt
                gt = torch.tensor([gt[0], gt[1], gt[2], gt[6], gt[7]])
                return pc, imu, gt
        # means that the index is in the last key sequence
        full_pc = np.load(os.path.join(self.path, f"{self.index[keys[-1]]}_pc.npy"))
        full_imu = np.load(os.path.join(self.path, f"{self.index[keys[-1]]}_imu.npy"))
        full_gt = np.load(os.path.join(self.path, f"{self.index[keys[-1]]}_gt.npy"))
        # get the index of the sample in the sequence
        idx_in_sequence = idx - keys[-1]
        # return the sample
        pc = torch.from_numpy(full_pc[idx_in_sequence:idx_in_sequence+self.frames]) # (T, P, 3)
        imu = torch.from_numpy(full_imu[idx_in_sequence:idx_in_sequence+self.frames]) # (T, 6)
        gt_np = full_gt[idx_in_sequence+self.frames-1] # (8,)
        if self.gt_as_rad:
            gt_np[6:8] = np.deg2rad(gt_np[6:8])
        if self.gt_limit is not None:
            gt_np = np.clip(gt_np, self.gt_limit[0], self.gt_limit[1])
        gt = torch.from_numpy(gt_np) # (8,)
        del full_pc, full_imu, full_gt
        gt = torch.tensor([gt[0], gt[1], gt[2], gt[6], gt[7]])
        return pc, imu, gt
    
    def gt_as_degree(self, gt: torch.Tensor):
        """
        Transforms the ground truth back to its original form.
        """
        return torch.tensor([gt[0], gt[1], gt[2], np.rad2deg(gt[3]), np.rad2deg(gt[4])])


class ImageDataset(Dataset):
    def __init__(self, data_folder: str, frames: int, gt_as_rad: bool = True, gt_limit: None | list = [-1, 1]):
        """
        Args:
            data_folder (string): Path to the image dataset's train/val folder.
                                  In each subfolder, there should be a labels.csv file and a folder for each sample. 
            frames (int): Number of frames in each sample.
            gt_as_rad (bool): Whether to return the ground truth as radians or not.
            gt_limit (None | list): If not None, the ground truth will be limited to the given range.
        """
        self.path = data_folder
        self.frames = frames
        self.gt_as_rad = gt_as_rad
        self.gt_limit = gt_limit
        self.theta_rounded_hist = []
        self.phi_rounded_hist = []
        self.x_rounded_hist = []
        self.y_rounded_hist = []
        self.z_rounded_hist = []
        # the keys represent the cumulative number of samples
        self.index = {}
        self.samples = 0
        print(f"Initializing dataset from '{self.path}'...")
        # loop through folders
        for file in os.listdir(self.path):
            filename = os.fsdecode(file)
            if filename.endswith("_gt.npy"): # load sequence set at once and not after another
                sequence_id = int(filename.split("_")[0])
                imgs = np.load(os.path.join(self.path, f"{sequence_id}_img.npy"))
                imus = np.load(os.path.join(self.path, f"{sequence_id}_imu.npy"))
                gts = np.load(os.path.join(self.path, f"{sequence_id}_gt.npy"))
                if imgs.shape[0] != imus.shape[0] or imgs.shape[0] != gts.shape[0]:
                    raise Exception(f"In sequence {sequence_id}, the number of images ({imgs.shape[0]}), IMU ({imus.shape[0]}) and GT ({gts.shape[0]}) do not match.")
                samples_in_imgs = imgs.shape[0] - self.frames + 1
                # add the sequence to the index
                self.index[self.samples] = sequence_id
                self.samples += samples_in_imgs
                # collect histogram data
                temp_x_rounded_hist = np.round(gts[:, 0], 1)
                temp_y_rounded_hist = np.round(gts[:, 1], 1)
                temp_z_rounded_hist = np.round(gts[:, 2], 1)
                temp_theta_rounded_hist = np.round(np.deg2rad(gts[:, 6]) if self.gt_as_rad else gts[:, 6], 1)
                temp_phi_rounded_hist = np.round(np.deg2rad(gts[:, 7]) if self.gt_as_rad else gts[:, 7], 1)
                if self.gt_limit is not None:
                    temp_x_rounded_hist = np.clip(temp_x_rounded_hist, self.gt_limit[0], self.gt_limit[1])
                    temp_y_rounded_hist = np.clip(temp_y_rounded_hist, self.gt_limit[0], self.gt_limit[1])
                    temp_z_rounded_hist = np.clip(temp_z_rounded_hist, self.gt_limit[0], self.gt_limit[1])
                    temp_theta_rounded_hist = np.clip(temp_theta_rounded_hist, self.gt_limit[0], self.gt_limit[1])
                    temp_phi_rounded_hist = np.clip(temp_phi_rounded_hist, self.gt_limit[0], self.gt_limit[1])
                self.x_rounded_hist += temp_x_rounded_hist.tolist()
                self.y_rounded_hist += temp_y_rounded_hist.tolist()
                self.z_rounded_hist += temp_z_rounded_hist.tolist()
                self.theta_rounded_hist += temp_theta_rounded_hist.tolist()
                self.phi_rounded_hist += temp_phi_rounded_hist.tolist()
                del imgs, imus, gts
        # create histograms
        self.x_rounded_hist = np.unique(self.x_rounded_hist, return_counts=True)
        self.y_rounded_hist = np.unique(self.y_rounded_hist, return_counts=True)
        self.z_rounded_hist = np.unique(self.z_rounded_hist, return_counts=True)
        self.theta_rounded_hist = np.unique(self.theta_rounded_hist, return_counts=True)
        self.phi_rounded_hist = np.unique(self.phi_rounded_hist, return_counts=True)
        print(f"Dataset initialized with {self.samples} samples.")
    

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        """
        Returns:
            imgs (torch.Tensor): The image data of the sample, shape (T, 1, H, W).
            imu (torch.Tensor): The IMU data of the sample, shape (6, T).
            gt (torch.Tensor): The ground truth data of the sample, shape (5,).
        """
        # find the sequence that contains the index
        keys = sorted(self.index.keys())
        for i, key in enumerate(keys):
            if idx < key:
                # means that the index is in previous key sequence
                full_imgs = np.load(os.path.join(self.path, f"{self.index[keys[i-1]]}_img.npy"))
                full_imu = np.load(os.path.join(self.path, f"{self.index[keys[i-1]]}_imu.npy"))
                full_gt = np.load(os.path.join(self.path, f"{self.index[keys[i-1]]}_gt.npy"))
                # get the index of the sample in the sequence
                idx_in_sequence = idx - keys[i-1]
                # return the sample
                imgs = torch.from_numpy(full_imgs[idx_in_sequence:idx_in_sequence+self.frames].astype(np.float32)) # (T, H, V)
                imu = torch.from_numpy(full_imu[idx_in_sequence:idx_in_sequence+self.frames]) # (T, 6)
                gt_np = full_gt[idx_in_sequence+self.frames-1] # (8,)
                if self.gt_as_rad:
                    gt_np[6:8] = np.deg2rad(gt_np[6:8])
                if self.gt_limit is not None:
                    gt_np = np.clip(gt_np, self.gt_limit[0], self.gt_limit[1])
                gt = torch.from_numpy(gt_np) # (8,)
                # remove variables from memory
                del full_imgs, full_imu, full_gt
                imgs = imgs.permute(0, 2, 1).unsqueeze(1) # (T, 1, V, H)
                gt = torch.tensor([gt[0], gt[1], gt[2], gt[6], gt[7]])
                return imgs, imu, gt
        # means that the index is in the last key sequence
        full_imgs = np.load(os.path.join(self.path, f"{self.index[keys[-1]]}_img.npy"))
        full_imu = np.load(os.path.join(self.path, f"{self.index[keys[-1]]}_imu.npy"))
        full_gt = np.load(os.path.join(self.path, f"{self.index[keys[-1]]}_gt.npy"))
        # get the index of the sample in the sequence
        idx_in_sequence = idx - keys[-1]
        # return the sample
        imgs = torch.from_numpy(full_imgs[idx_in_sequence:idx_in_sequence+self.frames].astype(np.float32)) # (T, H, V)
        imu = torch.from_numpy(full_imu[idx_in_sequence:idx_in_sequence+self.frames]) # (T, 6)
        gt_np = full_gt[idx_in_sequence+self.frames-1] # (8,)
        if self.gt_as_rad:
            gt_np[6:8] = np.deg2rad(gt_np[6:8])
        if self.gt_limit is not None:
            gt_np = np.clip(gt_np, self.gt_limit[0], self.gt_limit[1])
        gt = torch.from_numpy(gt_np) # (8,)
        del full_imgs, full_imu, full_gt
        imgs = imgs.permute(0, 2, 1).unsqueeze(1)  # (T, 1, V, H)
        gt = torch.tensor([gt[0], gt[1], gt[2], gt[6], gt[7]])
        return imgs, imu, gt

    def gt_as_degree(self, gt: torch.Tensor):
        """
        Transforms the ground truth back to its original form.
        """
        return torch.tensor([gt[0], gt[1], gt[2], np.rad2deg(gt[3]), np.rad2deg(gt[4])])
    

# ds = ImageDataset("C:/Users/bader/Desktop/SynthCave/data/4_staging/lidar1/depth_image/train", 2)

# max_phi, max_theta = 0, 0

# phis, thetas = [], []

# for i in range(len(ds)):
#     _, _, gt = ds[i]
#     phi, theta = gt[3].item(), gt[4].item()
#     if phi > max_phi:
#         max_phi = phi
#     if theta > max_theta:
#         max_theta = theta

#     phis.append(round(phi, 1))
#     thetas.append(round(theta, 1))
# print(max_phi, max_theta)
# # plot histogram
# import matplotlib.pyplot as plt
# plt.hist(phis, bins=100)
# plt.show()
# plt.hist(thetas, bins=100)
# plt.show()