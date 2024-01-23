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
        self.early_stop = False

    def __call__(self, val_loss):
        """
        Call the early stopping function.
        
        Parameters:
        - val_loss: float, validation loss.
        
        Returns:
        - early_stop: bool, whether to stop the training or not.
        """
        print(f"counter: {self.counter}")
        if self.val_loss_min is None:
            self.val_loss_min = val_loss
        elif val_loss > self.val_loss_min:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.val_loss_min = val_loss
            self.counter = 0
        return self.early_stop


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
    criterion = nn.MSELoss().to(device)

    # instantiate metrics
    mse = MeanSquaredError().to(device)

    # instantiate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=int(5), verbose=True, min_lr=1e-6)

    # Early stopping
    early_stopping = EarlyStopping(patience=10)

    # loop over epochs
    for epoch in range(epochs):

        # for graph models, we need to unpack one more element from the dataloader
        if model.dataset_type == "graph":
            # Training
            model.train()
            total_loss = 0.0
            for graph, edges, imu, gt in train_dl:
                graph, edges, imu, gt = graph.to(device).float(), edges.to(device).float(), imu.to(device).float(), gt.to(device).float()
                optimizer.zero_grad()
                outputs = model(graph, edges, imu)
                loss = criterion(outputs, gt)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                mse.update(outputs, gt)
            avg_loss = total_loss / len(train_dl)
            avg_mse_train = mse.compute().item()
            mse.reset()
            # Validation
            model.eval()
            with torch.no_grad():
                for graph, edges, imu, gt in val_dl:
                    graph, edges, imu, gt = graph.to(device).float(), edges.to(device).float(), imu.to(device).float(), gt.to(device).float()
                    outputs = model(graph, edges, imu)
                    mse.update(outputs, gt)
            avg_mse_val = mse.compute().item()
            mse.reset()
        
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
                mse.update(outputs, gt)
            avg_loss = total_loss / len(train_dl)
            avg_mse_train = mse.compute().item()
            mse.reset()
            # Validation
            model.eval()
            with torch.no_grad():
                for content, imu, gt in val_dl:
                    content, imu, gt = content.to(device).float(), imu.to(device).float(), gt.to(device).float()
                    outputs = model(content, imu)
                    mse.update(outputs, gt)
            avg_mse_val = mse.compute().item()
            mse.reset()

        # End of epoch
        scheduler.step(avg_mse_val)
        # log data and rount to 3 decimal places
        log.info(f"Epoch {epoch+1}/{epochs} | Loss: {round(avg_loss, 3)} | Train MSE: {round(avg_mse_train, 3)} | Val MSE: {round(avg_mse_val, 3)} | LR: {optimizer.param_groups[0]['lr']}")
        wandb.log({
            "loss": avg_loss,
            "train_mse": avg_mse_train,
            "val_mse": avg_mse_val,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "epoch": epoch+1
        })
        if early_stopping(avg_mse_val):
            log.info("Early stopping")
            break
    # save model
    torch.save(model.state_dict(), f"models/{wandb.run.name}.pt")


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
        "name": f"mv_sweep_{time.time()}",
        "metric": {"name": "val_mse", "goal": "maximize"},
        "method": "grid",
        "parameters": {
            "model_name": {"values": ["CNN", "TSViTcls", "ASTGCN", "NTU"]},
            "graph_dataset_path": {"values": [args.graph_dataset_path]},
            "image_dataset_path": {"values": [args.image_dataset_path]},
            "point_dataset_path": {"values": [args.point_dataset_path]},
            "epochs": {"values": [30]},
            "batch_size": {"values": [8]},
            "K": {"values": [2, 4, 8, 16]}
        }
    }
    sweep_id = wandb.sweep(sweep_configuration)

    # run the sweep
    wandb.agent(sweep_id, function=wandb_sweep)
