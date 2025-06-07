"""
Script to evaluate trained RAM model checkpoints on the test split.
This script loads multiple checkpoints and runs evaluation on the test dataset.
"""
import sys
from pathlib import Path
import torch
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from typing import List, Dict

from glimpse_attention_networks.models.ram_lit_module import RecurrentAttentionModel
from glimpse_attention_networks.datasets.svhn_datamodule import SVHNDataModule
from glimpse_attention_networks.visualization.attention_visualizer import AttentionVisualizer
from glimpse_attention_networks.utils.config import load_config


def get_experiment_dir(ckpt_path: str) -> Path:
    """
    Get the experiment directory from checkpoint path.
    Expected path format: .../results/{experiment_name}/{timestamp}/.../checkpoints/...
    """
    path = Path(ckpt_path)
    # Navigate up to the experiment directory (timestamp level)
    while path.parent.name != 'results' and path.parent != path:
        path = path.parent
    if path.parent == path:
        return Path("unknown_experiment")
    return path


def evaluate_checkpoint(ckpt_path: str, data_module: SVHNDataModule, config: DictConfig) -> dict:
    """
    Evaluate a trained RAM model checkpoint on the test split.
    
    Args:
        ckpt_path: Path to the model checkpoint
        data_module: Initialized data module to reuse
        config: Configuration dictionary
        
    Returns:
        dict: Dictionary containing test metrics
    """
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # Initialize model
    model = RecurrentAttentionModel.load_from_checkpoint(ckpt_path)
    model.eval()
    
    # Get experiment directory for logging
    experiment_dir = get_experiment_dir(ckpt_path)
    
    # Initialize trainer with experiment directory as log dir
    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=str(experiment_dir)
    )
    
    # Run evaluation
    test_results = trainer.test(model, datamodule=data_module)
    
    return test_results[0]  # Return first (and only) set of results


def evaluate_multiple_checkpoints(ckpt_paths: List[str]) -> Dict[str, dict]:
    """
    Evaluate multiple checkpoints and organize results by experiment name.
    
    Args:
        ckpt_paths: List of paths to model checkpoints
        
    Returns:
        Dict mapping experiment names to their test results
    """
    results = {}
    
    # Load config once
    config = load_config(config_name='train')
    
    # Initialize data module once
    data_module = hydra.utils.instantiate(config.datamodule)
    
    for ckpt_path in ckpt_paths:
        print(f"\nEvaluating checkpoint: {ckpt_path}")
        try:
            experiment_name = get_experiment_dir(ckpt_path).name
            test_metrics = evaluate_checkpoint(ckpt_path, data_module, config)
            results[experiment_name] = test_metrics
            print(f"Successfully evaluated {experiment_name}")
        except Exception as e:
            print(f"Error evaluating {ckpt_path}: {str(e)}")
            continue
    
    return results


if __name__ == "__main__":
    # List of checkpoint paths to evaluate
    checkpoint_paths = [
        "/mnt/c/users/victor/documents/projects/Glimpse Attention Networks/results/baseline_with_extra_train/2025-06-02_23-50-19/metrics/version_0/checkpoints/ram-epoch=04-val_accuracy=0.901.ckpt",
        "/mnt/c/users/victor/documents/projects/Glimpse Attention Networks/results/glimpse_3_baseline_with_extra_train_glimpse_size_8/2025-06-03_10-29-12/metrics/version_0/checkpoints/ram-epoch=08-val_accuracy=0.906.ckpt",
        "/mnt/c/users/victor/documents/projects/Glimpse Attention Networks/results/glimpse_3_baseline_with_extra_train_glimpse_size_12/2025-06-03_11-27-27/metrics/version_0/checkpoints/ram-epoch=02-val_accuracy=0.938.ckpt",
        "/mnt/c/users/victor/documents/projects/Glimpse Attention Networks/results/glimpse_3_baseline_with_extra_train_glimpse_size_16/2025-06-03_14-23-15/metrics/version_0/checkpoints/ram-epoch=02-val_accuracy=0.934.ckpt",
        "/mnt/c/users/victor/documents/projects/Glimpse Attention Networks/results/glimpse_6_baseline_with_extra_train_glimpse_size_8/2025-06-03_15-24-28/metrics/version_0/checkpoints/ram-epoch=00-val_accuracy=0.850.ckpt",
        "/mnt/c/users/victor/documents/projects/Glimpse Attention Networks/results/glimpse_6_baseline_with_extra_train_glimpse_size_12/2025-06-03_19-39-35/metrics/version_0/checkpoints/ram-epoch=01-val_accuracy=0.935.ckpt",
        "/mnt/c/users/victor/documents/projects/Glimpse Attention Networks/results/glimpse_6_baseline_with_extra_train_glimpse_size_16/2025-06-03_20-58-35/metrics/version_0/checkpoints/ram-epoch=00-val_accuracy=0.922.ckpt",
    ]
    
    # Evaluate all checkpoints
    results = evaluate_multiple_checkpoints(checkpoint_paths)
    
    # Print summary of results
    print("\n=== Test Results Summary ===")
    print("=" * 50)
    for experiment_name, metrics in results.items():
        print(f"\nExperiment: {experiment_name}")
        print("-" * 30)
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
        print("-" * 30) 