"""
train.py

Author: Tim Bader
Date: 19.01.2024

This script trains a PyTorch model using wandb sweep. It sets up the model, dataset, 
optimizer, loss function, metrics, and scheduler, and then trains the model over a 
number of epochs.

This script is intended to be run as a standalone program and requires the following command-line arguments:
- `-w` or `--wandb_project`: WandB project name
- `-g` or `--graph_dataset_path`: Path to the graph dataset
- `-i` or `--image_dataset_path`: Path to the image dataset
- `-p` or `--point_dataset_path`: Path to the point cloud dataset

Example:
python baselines/train.py -w "synthcave" -g "C:/Users/bader/Desktop/SynthCave/data/4_staging/lidar1/graph" -i "C:/Users/bader/Desktop/SynthCave/data/4_staging/lidar1/depth_image" -p "C:/Users/bader/Desktop/SynthCave/data/4_staging/lidar1/point_cloud" 
    
"""
import argparse
import wandb
import sys
import traceback
import logging
import time
import torch
import numpy as np
import torch.nn as nn
from torch.optim import AdamW
from torchmetrics import MeanSquaredError

from model.ASTGCN.ASTGCN import ASTGCN
from model.CNN.CNN import CNN
from model.PSTNet.PSTNet import NTU
from model.TSViTcls.TSViT import TSViTcls

from dataset.SynthCave import GraphDataset, ImageDataset, PointDataset


# format logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
generator = torch.Generator().manual_seed(seed)


class EarlyStopping:
    def __init__(self, patience=5):
        """
        Early stops the training if validation loss doesn't improve after a given patience.
        
        Parameters:
        - patience: int, optional, number of epochs to wait if no improvement and then stop the training.
        """
        self.patience = patience
        self.counter = 0
        self.val_mse = np.Inf

    def __call__(self, val_mse):
        """
        Call the early stopping function.
        
        Parameters:
        - val_mse: float, validation position MSE.
        
        Returns:
        - early_stop: bool, whether to stop the training or not.
        """
        if val_mse >= self.val_mse:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.counter = 0
            self.val_mse = val_mse
        return False
    

class DynamicWeightedMSELoss(nn.Module):
    def __init__(self, dataset):
        super(DynamicWeightedMSELoss, self).__init__()
        self.theta_steps, self.theta_counts = dataset.theta_rounded_hist # steps, counts
        self.total_thetas = sum(self.theta_counts) # total number of thetas
        self.phi_steps, self.phi_counts = dataset.phi_rounded_hist
        self.total_phis = sum(self.phi_counts)
        self.x_steps, self.x_counts = dataset.x_rounded_hist
        self.total_xs = sum(self.x_counts)
        self.y_steps, self.y_counts = dataset.y_rounded_hist
        self.total_ys = sum(self.y_counts)
        self.z_steps, self.z_counts = dataset.z_rounded_hist
        self.total_zs = sum(self.z_counts)

    def forward(self, input, target):
        # Calculate weights based on input values
        weights = self.calculate_weights(input).to(input.device)
        # Ensure that weights, input and target are the same size
        assert weights.size() == input.size() == target.size()
        # Calculate the weighted MSE loss
        loss = weights * (input - target) ** 2
        return loss.mean()

    def calculate_weights(self, input):
        B, _ = input.size()
        x_weight = torch.zeros(B)
        y_weight = torch.zeros(B)
        z_weight = torch.zeros(B)
        theta_weight = torch.zeros(B)
        phi_weight = torch.zeros(B)
        for i in range(B):
            x_weight[i] = self.calculate_single_weight(input[i][0], self.x_steps, self.x_counts, self.total_xs)
            y_weight[i] = self.calculate_single_weight(input[i][1], self.y_steps, self.y_counts, self.total_ys)
            z_weight[i] = self.calculate_single_weight(input[i][2], self.z_steps, self.z_counts, self.total_zs)
            theta_weight[i] = self.calculate_single_weight(input[i][3], self.theta_steps, self.theta_counts, self.total_thetas)
            phi_weight[i] = self.calculate_single_weight(input[i][4], self.phi_steps, self.phi_counts, self.total_phis)
        weights = torch.stack([x_weight, y_weight, z_weight, theta_weight, phi_weight], dim=1)
        return weights

    def calculate_single_weight(self, input, steps, counts, total):
        rounded_input = torch.round(input, decimals=1)
        if rounded_input in steps:
            index = steps.index(rounded_input)
            return torch.tensor(1 - counts[index] / total)
        else:
            return torch.tensor(1)
        

def get_model_from_name(model_name, K):
    """
    Get a model from a string name.
    
    Parameters:
    - model_name: str, name of the model.
    - K: int, number of frames.
    
    Returns:
    - model: PyTorch model.
    """
    if model_name == "CNN":
        return CNN(K=K)
    elif model_name == "TSViTcls":
        return TSViTcls(K=K)
    elif model_name == "ASTGCN":
        return ASTGCN(K=K)
    elif model_name == "NTU":
        return NTU(K=K)
    else:
        raise ValueError("Model name not supported")


def wandb_sweep():
    """
    WandB wrapper function for training.
    """
    try:
        train_model()
    except Exception as e:
        log.error(traceback.print_exc(), file=sys.stderr)
        raise e


def train_model():
    """
    Train a model on a dataset for a number of epochs and log the results to wandb.
    Used by wandb sweep.
    """
    wandb.init()

    # get hyperparameters
    model_name = wandb.config.model_name
    graph_dataset_path = wandb.config.graph_dataset_path
    image_dataset_path = wandb.config.image_dataset_path
    point_dataset_path = wandb.config.point_dataset_path
    epochs = wandb.config.epochs
    batch_size = wandb.config.batch_size
    K = wandb.config.K
    # instantiate device
    device = torch.device("cuda")

    # instantiate model
    model = get_model_from_name(model_name, K)
    assert model.dataset_type in ["graph", "image", "point"], "Model dataset type not supported"
    trainable_parameter_count = round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 3)
    model.to(device)

    log.info(f"Model: {model.__class__.__name__} | Trainable parameters: {trainable_parameter_count}M | Dataset type: {model.dataset_type}")
    log.info(f"K: {K} | Epochs: {epochs} | Batch size: {batch_size} | Device: {device}")

    # instantiate dataset
    if model.dataset_type == "graph":
        train_ds = GraphDataset(graph_dataset_path+"/train", frames=K)
        val_ds = GraphDataset(graph_dataset_path+"/val", frames=K)
    elif model.dataset_type == "image":
        train_ds = ImageDataset(image_dataset_path+"/train", frames=K)
        val_ds = ImageDataset(image_dataset_path+"/val", frames=K)
    elif model.dataset_type == "point":
        train_ds = PointDataset(point_dataset_path+"/train", frames=K)
        val_ds = PointDataset(point_dataset_path+"/val", frames=K)
    
    # instantiate dataloaders
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
        generator=generator
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
        generator=generator
    )

    # instantiate optimizer
    optimizer = AdamW(model.parameters(), lr=0.001)

    # instantiate loss function
    criterion = DynamicWeightedMSELoss(train_ds).to(device)

    # instantiate metrics
    x_mse = MeanSquaredError().to(device)
    y_mse = MeanSquaredError().to(device)
    z_mse = MeanSquaredError().to(device)
    theta_mse = MeanSquaredError().to(device)
    phi_mse = MeanSquaredError().to(device)

    # instantiate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=int(5), verbose=True, min_lr=1e-6)

    # Early stopping
    early_stopping = EarlyStopping(patience=10)

    # loop over epochs
    lowest_val_mse = np.Inf
    for epoch in range(epochs):

        # for graph models, we need to unpack one more element from the dataloader
        if model.dataset_type == "graph":
            # Training
            model.train()
            total_loss = 0.0
            for graph, edges, imu, gt in train_dl:
                graph, edges, imu, gt = graph.to(device).float(), edges.to(device).to(torch.int64), imu.to(device).float(), gt.to(device).float()
                optimizer.zero_grad()
                outputs = model(graph, edges, imu)
                loss = criterion(outputs, gt)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                x_mse.update(outputs[:, 0], gt[:, 0])
                y_mse.update(outputs[:, 1], gt[:, 1])
                z_mse.update(outputs[:, 2], gt[:, 2])
                theta_mse.update(outputs[:, 3], gt[:, 3])
                phi_mse.update(outputs[:, 4], gt[:, 4])
            avg_loss = total_loss / len(train_dl)
            avg_x_mse_train = x_mse.compute().item()
            avg_y_mse_train = y_mse.compute().item()
            avg_z_mse_train = z_mse.compute().item()
            avg_theta_mse_train = theta_mse.compute().item()
            avg_phi_mse_train = phi_mse.compute().item()
            x_mse.reset()
            y_mse.reset()
            z_mse.reset()
            theta_mse.reset()
            phi_mse.reset()
            # Validation
            model.eval()
            with torch.no_grad():
                for graph, edges, imu, gt in val_dl:
                    graph, edges, imu, gt = graph.to(device).float(), edges.to(device).to(torch.int64), imu.to(device).float(), gt.to(device).float()
                    outputs = model(graph, edges, imu)
                    x_mse.update(outputs[:, 0], gt[:, 0])
                    y_mse.update(outputs[:, 1], gt[:, 1])
                    z_mse.update(outputs[:, 2], gt[:, 2])
                    theta_mse.update(outputs[:, 3], gt[:, 3])
                    phi_mse.update(outputs[:, 4], gt[:, 4])
            avg_x_mse_val = x_mse.compute().item()
            avg_y_mse_val = y_mse.compute().item()
            avg_z_mse_val = z_mse.compute().item()
            avg_theta_mse_val = theta_mse.compute().item()
            avg_phi_mse_val = phi_mse.compute().item()
            x_mse.reset()
            y_mse.reset()
            z_mse.reset()
            theta_mse.reset()
            phi_mse.reset()
        
        #  model.dataset_type == "image" or model.dataset_type == "point":
        else:
            # Training
            model.train()
            total_loss = 0.0
            for content, imu, gt in train_dl:
                content, imu, gt = content.to(device).float(), imu.to(device).float(), gt.to(device).float()
                optimizer.zero_grad()
                outputs = model(content, imu)
                loss = criterion(outputs, gt)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                x_mse.update(outputs[:, 0], gt[:, 0])
                y_mse.update(outputs[:, 1], gt[:, 1])
                z_mse.update(outputs[:, 2], gt[:, 2])
                theta_mse.update(outputs[:, 3], gt[:, 3])
                phi_mse.update(outputs[:, 4], gt[:, 4])
            avg_loss = total_loss / len(train_dl)
            avg_x_mse_train = x_mse.compute().item()
            avg_y_mse_train = y_mse.compute().item()
            avg_z_mse_train = z_mse.compute().item()
            avg_theta_mse_train = theta_mse.compute().item()
            avg_phi_mse_train = phi_mse.compute().item()
            x_mse.reset()
            y_mse.reset()
            z_mse.reset()
            theta_mse.reset()
            phi_mse.reset()
            # Validation
            model.eval()
            with torch.no_grad():
                for content, imu, gt in val_dl:
                    content, imu, gt = content.to(device).float(), imu.to(device).float(), gt.to(device).float()
                    outputs = model(content, imu)
                    x_mse.update(outputs[:, 0], gt[:, 0])
                    y_mse.update(outputs[:, 1], gt[:, 1])
                    z_mse.update(outputs[:, 2], gt[:, 2])
                    theta_mse.update(outputs[:, 3], gt[:, 3])
                    phi_mse.update(outputs[:, 4], gt[:, 4])
            avg_x_mse_val = x_mse.compute().item()
            avg_y_mse_val = y_mse.compute().item()
            avg_z_mse_val = z_mse.compute().item()
            avg_theta_mse_val = theta_mse.compute().item()
            avg_phi_mse_val = phi_mse.compute().item()
            x_mse.reset()
            y_mse.reset()
            z_mse.reset()
            theta_mse.reset()
            phi_mse.reset()

        # End of epoch
        # log data and rount to 3 decimal places
        log.info(f"Epoch {epoch+1}/{epochs} | Loss: {round(avg_loss, 3)} | Learning rate: {round(optimizer.param_groups[0]['lr'], 6)}" +
                 f" | Train X MSE: {round(avg_x_mse_train, 3)} | Train Y MSE: {round(avg_y_mse_train, 3)} | Train Z MSE: {round(avg_z_mse_train, 3)}" +
                 f" | Train Theta MSE: {round(avg_theta_mse_train, 3)} | Train Phi MSE: {round(avg_phi_mse_train, 3)}" +
                 f" | AVG Pos MSE Train: {round((avg_x_mse_train+avg_y_mse_train+avg_z_mse_train)/3, 3)} | AVG Rot MSE Train: {round((avg_theta_mse_train+avg_phi_mse_train)/2, 3)}" +
                 f" | Val X MSE: {round(avg_x_mse_val, 3)} | Val Y MSE: {round(avg_y_mse_val, 3)} | Val Z MSE: {round(avg_z_mse_val, 3)}" + 
                 f" | Val Theta MSE: {round(avg_theta_mse_val, 3)} | Val Phi MSE: {round(avg_phi_mse_val, 3)}" + 
                 f" | AVG Pos MSE Val: {round((avg_x_mse_val+avg_y_mse_val+avg_z_mse_val)/3, 3)} | AVG Rot MSE Val: {round((avg_theta_mse_val+avg_phi_mse_val)/2, 3)}")
        wandb.log({
            "epoch": epoch+1,
            "loss": avg_loss,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "train_x_mse": avg_x_mse_train,
            "train_y_mse": avg_y_mse_train,
            "train_z_mse": avg_z_mse_train,
            "train_theta_mse": avg_theta_mse_train,
            "train_phi_mse": avg_phi_mse_train,
            "train_avg_pos_mse": (avg_x_mse_train+avg_y_mse_train+avg_z_mse_train)/3,
            "train_avg_rot_mse": (avg_theta_mse_train+avg_phi_mse_train)/2,
            "val_x_mse": avg_x_mse_val,
            "val_y_mse": avg_y_mse_val,
            "val_z_mse": avg_z_mse_val,
            "val_theta_mse": avg_theta_mse_val,
            "val_phi_mse": avg_phi_mse_val
        })
        avg_val_mse = (avg_x_mse_val+avg_y_mse_val+avg_z_mse_val+avg_theta_mse_val+avg_phi_mse_val)/5
        # save model
        if avg_val_mse < lowest_val_mse:
            torch.save(model.state_dict(), f"models/{wandb.run.name}.pt")
            lowest_val_mse = avg_val_mse
        if early_stopping(avg_val_mse):
            log.info("Early stopping")
            break
        scheduler.step(avg_val_mse)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--wandb_project", type=str, required=True, help="WandB project name")
    parser.add_argument("-g", "--graph_dataset_path", type=str, required=True, help="Path to the graph dataset")
    parser.add_argument("-i", "--image_dataset_path", type=str, required=True, help="Path to the image dataset")
    parser.add_argument("-p", "--point_dataset_path", type=str, required=True, help="Path to the point cloud dataset")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is not available"

    sweep_configuration = {
        "project": args.wandb_project,
        "name": f"train_sweep_{time.time()}",
        "metric": {"name": "val_mse", "goal": "minimize"},
        "method": "grid",
        "parameters": {
            "model_name": {"values": ["TSViTcls", "CNN", "ASTGCN", "NTU"]},
            "graph_dataset_path": {"values": [args.graph_dataset_path]},
            "image_dataset_path": {"values": [args.image_dataset_path]},
            "point_dataset_path": {"values": [args.point_dataset_path]},
            "epochs": {"values": [30]},
            "batch_size": {"values": [8]},
            "K": {"values": [16, 8, 4, 2]}
        }
    }
    sweep_id = wandb.sweep(sweep_configuration)

    # run the sweep
    wandb.agent(sweep_id, function=wandb_sweep)
