"""
eval.py

Author: Tim Bader
Date: 29.01.2024

This script evaluates PyTorch models using wandb sweep.

Run script with:

python eval.py

The input parameters must be configured at the bottom of this file.

"""
import wandb
import sys
import traceback
import logging
import time
import torch
import numpy as np
from torchmetrics import MeanSquaredError
from torchmetrics import MeanAbsoluteError

from model.ASTGCN.ASTGCN import ASTGCN
from model.CNN.CNN import CNN
from model.PSTNet.PSTNet import NTU
from model.TSViTcls.TSViT import TSViTcls
from model.ZERO.ZERO import ZERO
from model.RANDOM.RANDOM import RANDOM
from model.ICP.ICP import ICP

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


class RootMeanSquaredError(MeanSquaredError):
    def compute(self):
        return torch.sqrt(super().compute())



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
    elif model_name == "ZERO":
        return ZERO(K=K)
    elif model_name == "RANDOM":
        return RANDOM(K=K)
    elif model_name == "ICP":
        return ICP(K=K)
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
    model_config = wandb.config.model_config
    model_name = model_config["model_name"]
    model_path = model_config["path"]
    K = model_config["K"]
    graph_dataset_path = wandb.config.graph_dataset_path
    image_dataset_path = wandb.config.image_dataset_path
    point_dataset_path = wandb.config.point_dataset_path
    batch_size = wandb.config.batch_size
    # instantiate device
    device = torch.device("cuda")

    # instantiate model
    model = get_model_from_name(model_name, K)
    assert model.dataset_type in ["graph", "image", "point"], "Model dataset type not supported"
    trainable_parameter_count = round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 3)
    if model_path != "":
        model.load_state_dict(torch.load(model_path))
    model.to(device)

    log.info(f"Model: {model.__class__.__name__} | Trainable parameters: {trainable_parameter_count}M | Dataset type: {model.dataset_type}")

    # instantiate dataset
    if model.dataset_type == "graph":
        test_ds = GraphDataset(graph_dataset_path+"/test", frames=K, return_seq_name=True)
    elif model.dataset_type == "image":
        test_ds = ImageDataset(image_dataset_path+"/test", frames=K, return_seq_name=True)
    elif model.dataset_type == "point":
        test_ds = PointDataset(point_dataset_path+"/test", frames=K, return_seq_name=True)
    
    # instantiate dataloaders
    test_dl = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
        generator=generator
    )

    # instantiate metrics
    x_mse, x_rmse, x_ame = MeanSquaredError().to(device), RootMeanSquaredError().to(device), MeanAbsoluteError().to(device)
    y_mse, y_rmse, y_ame = MeanSquaredError().to(device), RootMeanSquaredError().to(device), MeanAbsoluteError().to(device)
    z_mse, z_rmse, z_ame = MeanSquaredError().to(device), RootMeanSquaredError().to(device), MeanAbsoluteError().to(device)
    theta_mse, theta_rmse, theta_ame = MeanSquaredError().to(device), RootMeanSquaredError().to(device), MeanAbsoluteError().to(device)
    phi_mse, phi_rmse, phi_ame = MeanSquaredError().to(device), RootMeanSquaredError().to(device), MeanAbsoluteError().to(device)

    # track metrics
    metrics = {}

    # for graph models, we need to unpack one more element from the dataloader
    if model.dataset_type == "graph":
        model.eval()
        with torch.no_grad():
            for graph, edges, imu, gt, sequence_name in test_dl:
                graph, edges, imu, gt = graph.to(device).float(), edges.to(device).to(torch.int64), imu.to(device).float(), gt.to(device).float()
                outputs = model(graph, edges, imu)
                x_mse.update(outputs[:, 0], gt[:, 0])
                y_mse.update(outputs[:, 1], gt[:, 1])
                z_mse.update(outputs[:, 2], gt[:, 2])
                theta_mse.update(outputs[:, 3], gt[:, 3])
                phi_mse.update(outputs[:, 4], gt[:, 4])
                avg_x_mse_test = x_mse.compute().item()
                avg_y_mse_test = y_mse.compute().item()
                avg_z_mse_test = z_mse.compute().item()
                avg_theta_mse_test = theta_mse.compute().item()
                avg_phi_mse_test = phi_mse.compute().item()
                x_mse.reset()
                y_mse.reset()
                z_mse.reset()
                theta_mse.reset()
                phi_mse.reset()
                x_rmse.update(outputs[:, 0], gt[:, 0])
                y_rmse.update(outputs[:, 1], gt[:, 1])
                z_rmse.update(outputs[:, 2], gt[:, 2])
                theta_rmse.update(outputs[:, 3], gt[:, 3])
                phi_rmse.update(outputs[:, 4], gt[:, 4])
                avg_x_rmse_test = x_rmse.compute().item()
                avg_y_rmse_test = y_rmse.compute().item()
                avg_z_rmse_test = z_rmse.compute().item()
                avg_theta_rmse_test = theta_rmse.compute().item()
                avg_phi_rmse_test = phi_rmse.compute().item()
                x_rmse.reset()
                y_rmse.reset()
                z_rmse.reset()
                theta_rmse.reset()
                phi_rmse.reset()
                x_ame.update(outputs[:, 0], gt[:, 0])
                y_ame.update(outputs[:, 1], gt[:, 1])
                z_ame.update(outputs[:, 2], gt[:, 2])
                theta_ame.update(outputs[:, 3], gt[:, 3])
                phi_ame.update(outputs[:, 4], gt[:, 4])
                avg_x_ame_test = x_ame.compute().item()
                avg_y_ame_test = y_ame.compute().item()
                avg_z_ame_test = z_ame.compute().item()
                avg_theta_ame_test = theta_ame.compute().item()
                avg_phi_ame_test = phi_ame.compute().item()
                x_ame.reset()
                y_ame.reset()
                z_ame.reset()
                theta_ame.reset()
                phi_ame.reset()

                sequence_name = sequence_name[0]
                if sequence_name in metrics:
                    metrics[sequence_name]["x_mse"].append(avg_x_mse_test)
                    metrics[sequence_name]["y_mse"].append(avg_y_mse_test)
                    metrics[sequence_name]["z_mse"].append(avg_z_mse_test)
                    metrics[sequence_name]["theta_mse"].append(avg_theta_mse_test)
                    metrics[sequence_name]["phi_mse"].append(avg_phi_mse_test)
                    metrics[sequence_name]["x_rmse"].append(avg_x_rmse_test)
                    metrics[sequence_name]["y_rmse"].append(avg_y_rmse_test)
                    metrics[sequence_name]["z_rmse"].append(avg_z_rmse_test)
                    metrics[sequence_name]["theta_rmse"].append(avg_theta_rmse_test)
                    metrics[sequence_name]["phi_rmse"].append(avg_phi_rmse_test)
                    metrics[sequence_name]["x_ame"].append(avg_x_ame_test)
                    metrics[sequence_name]["y_ame"].append(avg_y_ame_test)
                    metrics[sequence_name]["z_ame"].append(avg_z_ame_test)
                    metrics[sequence_name]["theta_ame"].append(avg_theta_ame_test)
                    metrics[sequence_name]["phi_ame"].append(avg_phi_ame_test)
                else:
                    metrics[sequence_name] = {
                        "x_mse": [avg_x_mse_test],
                        "y_mse": [avg_y_mse_test],
                        "z_mse": [avg_z_mse_test],
                        "theta_mse": [avg_theta_mse_test],
                        "phi_mse": [avg_phi_mse_test],
                        "x_rmse": [avg_x_rmse_test],
                        "y_rmse": [avg_y_rmse_test],
                        "z_rmse": [avg_z_rmse_test],
                        "theta_rmse": [avg_theta_rmse_test],
                        "phi_rmse": [avg_phi_rmse_test],
                        "x_ame": [avg_x_ame_test],
                        "y_ame": [avg_y_ame_test],
                        "z_ame": [avg_z_ame_test],
                        "theta_ame": [avg_theta_ame_test],
                        "phi_ame": [avg_phi_ame_test]
                    }
    
    #  model.dataset_type == "image" or model.dataset_type == "point":
    else:
        # Validation
        model.eval()
        with torch.no_grad():
            for content, imu, gt, sequence_name in test_dl:
                content, imu, gt = content.to(device).float(), imu.to(device).float(), gt.to(device).float()
                outputs = model(content, imu)
                x_mse.update(outputs[:, 0], gt[:, 0])
                y_mse.update(outputs[:, 1], gt[:, 1])
                z_mse.update(outputs[:, 2], gt[:, 2])
                theta_mse.update(outputs[:, 3], gt[:, 3])
                phi_mse.update(outputs[:, 4], gt[:, 4])
                avg_x_mse_test = x_mse.compute().item()
                avg_y_mse_test = y_mse.compute().item()
                avg_z_mse_test = z_mse.compute().item()
                avg_theta_mse_test = theta_mse.compute().item()
                avg_phi_mse_test = phi_mse.compute().item()
                x_mse.reset()
                y_mse.reset()
                z_mse.reset()
                theta_mse.reset()
                phi_mse.reset()
                x_rmse.update(outputs[:, 0], gt[:, 0])
                y_rmse.update(outputs[:, 1], gt[:, 1])
                z_rmse.update(outputs[:, 2], gt[:, 2])
                theta_rmse.update(outputs[:, 3], gt[:, 3])
                phi_rmse.update(outputs[:, 4], gt[:, 4])
                avg_x_rmse_test = x_rmse.compute().item()
                avg_y_rmse_test = y_rmse.compute().item()
                avg_z_rmse_test = z_rmse.compute().item()
                avg_theta_rmse_test = theta_rmse.compute().item()
                avg_phi_rmse_test = phi_rmse.compute().item()
                x_rmse.reset()
                y_rmse.reset()
                z_rmse.reset()
                theta_rmse.reset()
                phi_rmse.reset()
                x_ame.update(outputs[:, 0], gt[:, 0])
                y_ame.update(outputs[:, 1], gt[:, 1])
                z_ame.update(outputs[:, 2], gt[:, 2])
                theta_ame.update(outputs[:, 3], gt[:, 3])
                phi_ame.update(outputs[:, 4], gt[:, 4])
                avg_x_ame_test = x_ame.compute().item()
                avg_y_ame_test = y_ame.compute().item()
                avg_z_ame_test = z_ame.compute().item()
                avg_theta_ame_test = theta_ame.compute().item()
                avg_phi_ame_test = phi_ame.compute().item()
                x_ame.reset()
                y_ame.reset()
                z_ame.reset()
                theta_ame.reset()
                phi_ame.reset()

                sequence_name = sequence_name[0]
                if sequence_name in metrics:
                    metrics[sequence_name]["x_mse"].append(avg_x_mse_test)
                    metrics[sequence_name]["y_mse"].append(avg_y_mse_test)
                    metrics[sequence_name]["z_mse"].append(avg_z_mse_test)
                    metrics[sequence_name]["theta_mse"].append(avg_theta_mse_test)
                    metrics[sequence_name]["phi_mse"].append(avg_phi_mse_test)
                    metrics[sequence_name]["x_rmse"].append(avg_x_rmse_test)
                    metrics[sequence_name]["y_rmse"].append(avg_y_rmse_test)
                    metrics[sequence_name]["z_rmse"].append(avg_z_rmse_test)
                    metrics[sequence_name]["theta_rmse"].append(avg_theta_rmse_test)
                    metrics[sequence_name]["phi_rmse"].append(avg_phi_rmse_test)
                    metrics[sequence_name]["x_ame"].append(avg_x_ame_test)
                    metrics[sequence_name]["y_ame"].append(avg_y_ame_test)
                    metrics[sequence_name]["z_ame"].append(avg_z_ame_test)
                    metrics[sequence_name]["theta_ame"].append(avg_theta_ame_test)
                    metrics[sequence_name]["phi_ame"].append(avg_phi_ame_test)
                else:
                    metrics[sequence_name] = {
                        "x_mse": [avg_x_mse_test],
                        "y_mse": [avg_y_mse_test],
                        "z_mse": [avg_z_mse_test],
                        "theta_mse": [avg_theta_mse_test],
                        "phi_mse": [avg_phi_mse_test],
                        "x_rmse": [avg_x_rmse_test],
                        "y_rmse": [avg_y_rmse_test],
                        "z_rmse": [avg_z_rmse_test],
                        "theta_rmse": [avg_theta_rmse_test],
                        "phi_rmse": [avg_phi_rmse_test],
                        "x_ame": [avg_x_ame_test],
                        "y_ame": [avg_y_ame_test],
                        "z_ame": [avg_z_ame_test],
                        "theta_ame": [avg_theta_ame_test],
                        "phi_ame": [avg_phi_ame_test]
                    }

    result_dict = {}
    for sequence_name in metrics:
        result_dict[sequence_name] = {
            "x_mse": np.mean(metrics[sequence_name]["x_mse"]),
            "y_mse": np.mean(metrics[sequence_name]["y_mse"]),
            "z_mse": np.mean(metrics[sequence_name]["z_mse"]),
            "2d_pos_mse": np.mean([np.mean(metrics[sequence_name]["x_mse"]), np.mean(metrics[sequence_name]["z_mse"])]),
            "pos_mse": np.mean([np.mean(metrics[sequence_name]["x_mse"]), np.mean(metrics[sequence_name]["y_mse"]), np.mean(metrics[sequence_name]["z_mse"])]),
            "theta_mse": np.mean(metrics[sequence_name]["theta_mse"]),
            "phi_mse": np.mean(metrics[sequence_name]["phi_mse"]),
            "rot_mse": np.mean([np.mean(metrics[sequence_name]["theta_mse"]), np.mean(metrics[sequence_name]["phi_mse"])]),
            "mse": np.mean([np.mean(metrics[sequence_name]["x_mse"]), np.mean(metrics[sequence_name]["y_mse"]), np.mean(metrics[sequence_name]["z_mse"]), np.mean(metrics[sequence_name]["theta_mse"]), np.mean(metrics[sequence_name]["phi_mse"])]),
            "x_rmse": np.mean(metrics[sequence_name]["x_rmse"]),
            "y_rmse": np.mean(metrics[sequence_name]["y_rmse"]),
            "z_rmse": np.mean(metrics[sequence_name]["z_rmse"]),
            "2d_pos_rmse": np.mean([np.mean(metrics[sequence_name]["x_rmse"]), np.mean(metrics[sequence_name]["z_rmse"])]),
            "pos_rmse": np.mean([np.mean(metrics[sequence_name]["x_rmse"]), np.mean(metrics[sequence_name]["y_rmse"]), np.mean(metrics[sequence_name]["z_rmse"])]),
            "theta_rmse": np.mean(metrics[sequence_name]["theta_rmse"]),
            "phi_rmse": np.mean(metrics[sequence_name]["phi_rmse"]),
            "rot_rmse": np.mean([np.mean(metrics[sequence_name]["theta_rmse"]), np.mean(metrics[sequence_name]["phi_rmse"])]),
            "rmse": np.mean([np.mean(metrics[sequence_name]["x_rmse"]), np.mean(metrics[sequence_name]["y_rmse"]), np.mean(metrics[sequence_name]["z_rmse"]), np.mean(metrics[sequence_name]["theta_rmse"]), np.mean(metrics[sequence_name]["phi_rmse"])]),
            "x_ame": np.mean(metrics[sequence_name]["x_ame"]),
            "y_ame": np.mean(metrics[sequence_name]["y_ame"]),
            "z_ame": np.mean(metrics[sequence_name]["z_ame"]),
            "2d_pos_ame": np.mean([np.mean(metrics[sequence_name]["x_ame"]), np.mean(metrics[sequence_name]["z_ame"])]),
            "pos_ame": np.mean([np.mean(metrics[sequence_name]["x_ame"]), np.mean(metrics[sequence_name]["y_ame"]), np.mean(metrics[sequence_name]["z_ame"])]),
            "theta_ame": np.mean(metrics[sequence_name]["theta_ame"]),
            "phi_ame": np.mean(metrics[sequence_name]["phi_ame"]),
            "rot_ame": np.mean([np.mean(metrics[sequence_name]["theta_ame"]), np.mean(metrics[sequence_name]["phi_ame"])]),
            "ame": np.mean([np.mean(metrics[sequence_name]["x_ame"]), np.mean(metrics[sequence_name]["y_ame"]), np.mean(metrics[sequence_name]["z_ame"]), np.mean(metrics[sequence_name]["theta_ame"]), np.mean(metrics[sequence_name]["phi_ame"])])
        }
    result_dict["avg"] = {
        "x_mse": np.mean([result_dict[sequence_name]["x_mse"] for sequence_name in result_dict]),
        "y_mse": np.mean([result_dict[sequence_name]["y_mse"] for sequence_name in result_dict]),
        "z_mse": np.mean([result_dict[sequence_name]["z_mse"] for sequence_name in result_dict]),
        "2d_pos_mse": np.mean([result_dict[sequence_name]["2d_pos_mse"] for sequence_name in result_dict]),
        "pos_mse": np.mean([result_dict[sequence_name]["pos_mse"] for sequence_name in result_dict]),
        "theta_mse": np.mean([result_dict[sequence_name]["theta_mse"] for sequence_name in result_dict]),
        "phi_mse": np.mean([result_dict[sequence_name]["phi_mse"] for sequence_name in result_dict]),
        "rot_mse": np.mean([result_dict[sequence_name]["rot_mse"] for sequence_name in result_dict]),
        "mse": np.mean([result_dict[sequence_name]["mse"] for sequence_name in result_dict]),
        "x_rmse": np.mean([result_dict[sequence_name]["x_rmse"] for sequence_name in result_dict]),
        "y_rmse": np.mean([result_dict[sequence_name]["y_rmse"] for sequence_name in result_dict]),
        "z_rmse": np.mean([result_dict[sequence_name]["z_rmse"] for sequence_name in result_dict]),
        "2d_pos_rmse": np.mean([result_dict[sequence_name]["2d_pos_rmse"] for sequence_name in result_dict]),
        "pos_rmse": np.mean([result_dict[sequence_name]["pos_rmse"] for sequence_name in result_dict]),
        "theta_rmse": np.mean([result_dict[sequence_name]["theta_rmse"] for sequence_name in result_dict]),
        "phi_rmse": np.mean([result_dict[sequence_name]["phi_rmse"] for sequence_name in result_dict]),
        "rot_rmse": np.mean([result_dict[sequence_name]["rot_rmse"] for sequence_name in result_dict]),
        "rmse": np.mean([result_dict[sequence_name]["rmse"] for sequence_name in result_dict]),
        "x_ame": np.mean([result_dict[sequence_name]["x_ame"] for sequence_name in result_dict]),
        "y_ame": np.mean([result_dict[sequence_name]["y_ame"] for sequence_name in result_dict]),
        "z_ame": np.mean([result_dict[sequence_name]["z_ame"] for sequence_name in result_dict]),
        "2d_pos_ame": np.mean([result_dict[sequence_name]["2d_pos_ame"] for sequence_name in result_dict]),
        "pos_ame": np.mean([result_dict[sequence_name]["pos_ame"] for sequence_name in result_dict]),
        "theta_ame": np.mean([result_dict[sequence_name]["theta_ame"] for sequence_name in result_dict]),
        "phi_ame": np.mean([result_dict[sequence_name]["phi_ame"] for sequence_name in result_dict]),
        "rot_ame": np.mean([result_dict[sequence_name]["rot_ame"] for sequence_name in result_dict]),
        "ame": np.mean([result_dict[sequence_name]["ame"] for sequence_name in result_dict])
    }
    result_string = ""
    for sequence_name in result_dict:
        result_string += f"{sequence_name}: {round(result_dict[sequence_name]['mse'], 4)}\n"

    log.info(result_string)
             
    wandb.log(result_dict)


if __name__ == "__main__":

    assert torch.cuda.is_available(), "CUDA is not available"

    sweep_configuration = {
        "project": "synthcave",
        "name": f"eval_sweep_{time.time()}",
        "metric": {"name": "mse", "goal": "minimize"},
        "method": "grid",
        "parameters": {
            "model_config": {
                "values": [
                    {"model_name": "ICP", "K": 2, "path": ""},
                    {"model_name": "RANDOM", "K": -1, "path": ""},
                    {"model_name": "ZERO", "K": -1, "path": ""},
                    {"model_name": "TSViTcls", "K": 4, "path": "C:/Users/bader/Desktop/SynthCave/models/worthy-sweep-9.pt"}, 
                    {"model_name": "CNN", "K": 2, "path": "C:/Users/bader/Desktop/SynthCave/models/worthy-sweep-14.pt"},
                    {"model_name": "ASTGCN", "K": 4, "path": "C:/Users/bader/Desktop/SynthCave/models/swift-sweep-11.pt"},
                    {"model_name": "NTU", "K": 4, "path": "C:/Users/bader/Desktop/SynthCave/models/likely-sweep-12.pt"}
                    ]},
            "graph_dataset_path": {"values": ["C:/Users/bader/Desktop/SynthCave/data/4_staging/lidar1/graph"]},
            "image_dataset_path": {"values": ["C:/Users/bader/Desktop/SynthCave/data/4_staging/lidar1/depth_image"]},
            "point_dataset_path": {"values": ["C:/Users/bader/Desktop/SynthCave/data/4_staging/lidar1/point_cloud"]},
            "batch_size": {"values": [1]} # bs=1 is important, otherwise sequence names cannot be mapped
        }
    }
    sweep_id = wandb.sweep(sweep_configuration)

    # run the sweep
    wandb.agent(sweep_id, function=wandb_sweep)
