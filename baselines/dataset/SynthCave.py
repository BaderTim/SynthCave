import os
import torch
import numpy as np
from torch.utils.data import Dataset

class GraphDataset(Dataset):
    def __init__(self, data_folder: str, frames: int):
        """
        Args:
            data_folder (string): Path to the graph dataset's train/val folder.
                                  In each subfolder, there should be a labels.csv file and a folder for each sample. 
            frames (int): Number of frames in each sample.
        """
        self.path = data_folder
        self.frames = frames
        # the keys represent the cumulative number of samples
        self.index = {}
        self.samples = 0
        print(f"Initializing dataset from '{self.path}'...")
        # loop through folders
        for file in os.listdir(self.path):
            filename = os.fsdecode(file)
            if filename.endswith("graph.npy"):
                sequence_id = int(filename.split("_")[0])
                graph = np.load(os.path.join(self.path, filename))
                samples_in_graph = graph.shape[0] - self.frames + 1
                self.index[self.samples] = sequence_id
                self.samples += samples_in_graph
        # remove variables from memory
        del graph
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
                imu = torch.from_numpy(full_imu[idx_in_sequence:idx_in_sequence+self.frames-1]) # (T, 6)
                gt = torch.from_numpy(full_gt[idx_in_sequence+self.frames]) # (8,)
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
        imu = torch.from_numpy(full_imu[idx_in_sequence:idx_in_sequence+self.frames]) # (T, 6)
        gt = torch.from_numpy(full_gt[idx_in_sequence+self.frames-1]) # (8,)
        del full_graph, full_edges, full_imu, full_gt
        graph = graph.permute(1, 2, 0)
        edges = edges[0].permute(1, 0)
        gt = torch.tensor([gt[0], gt[1], gt[2], gt[6], gt[7]])
        return graph, edges, imu, gt
            

class PointDataset(Dataset):
    def __init__(self, data_folder: str, frames: int):
        """
        Args:
            data_folder (string): Path to the point cloud dataset's train/val folder.
                                  In each subfolder, there should be a labels.csv file and a folder for each sample. 
            frames (int): Number of frames in each sample.
        """
        self.path = data_folder
        self.frames = frames
        # the keys represent the cumulative number of samples
        self.index = {}
        self.samples = 0
        print(f"Initializing dataset from '{self.path}'...")
        # loop through folders
        for file in os.listdir(self.path):
            filename = os.fsdecode(file)
            if filename.endswith("pc.npy"):
                sequence_id = int(filename.split("_")[0])
                pc = np.load(os.path.join(self.path, filename))
                samples_in_pc = pc.shape[0] - self.frames + 1
                self.index[self.samples] = sequence_id
                self.samples += samples_in_pc
        # remove variables from memory
        del pc
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
                gt = torch.from_numpy(full_gt[idx_in_sequence+self.frames-1]) # (8,)
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
        gt = torch.from_numpy(full_gt[idx_in_sequence+self.frames-1]) # (8,)
        del full_pc, full_imu, full_gt
        gt = torch.tensor([gt[0], gt[1], gt[2], gt[6], gt[7]])
        return pc, imu, gt
    

class ImageDataset(Dataset):
    def __init__(self, data_folder: str, frames: int):
        """
        Args:
            data_folder (string): Path to the image dataset's train/val folder.
                                  In each subfolder, there should be a labels.csv file and a folder for each sample. 
            frames (int): Number of frames in each sample.
        """
        self.path = data_folder
        self.frames = frames
        # the keys represent the cumulative number of samples
        self.index = {}
        self.samples = 0
        print(f"Initializing dataset from '{self.path}'...")
        # loop through folders
        for file in os.listdir(self.path):
            filename = os.fsdecode(file)
            if filename.endswith("img.npy"):
                sequence_id = int(filename.split("_")[0])
                imgs = np.load(os.path.join(self.path, filename))
                samples_in_imgs = imgs.shape[0] - self.frames + 1
                self.index[self.samples] = sequence_id
                self.samples += samples_in_imgs
        # remove variables from memory
        del imgs
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
                gt = torch.from_numpy(full_gt[idx_in_sequence+self.frames-1]) # (8,)
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
        gt = torch.from_numpy(full_gt[idx_in_sequence+self.frames-1]) # (8,)
        del full_imgs, full_imu, full_gt
        imgs = imgs.permute(0, 2, 1).unsqueeze(1)  # (T, 1, V, H)
        gt = torch.tensor([gt[0], gt[1], gt[2], gt[6], gt[7]])
        return imgs, imu, gt
