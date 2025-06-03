"""
Script to visualize attention patterns from a trained RAM model checkpoint.
This script loads a checkpoint, runs inference on test samples, and visualizes
the attention patterns for each sample.
"""
import sys
from pathlib import Path
import torch
import hydra
import pytorch_lightning as pl
from omegaconf import OmegaConf

from glimpse_attention_networks.models.ram_lit_module import RecurrentAttentionModel
from glimpse_attention_networks.datasets.svhn_datamodule import SVHNDataModule
from glimpse_attention_networks.visualization.attention_visualizer import AttentionVisualizer
from glimpse_attention_networks.utils.config import load_config


def visualize_test_samples(model, data_module, num_samples=10, save_dir=None):
    """Visualize attention patterns for test samples."""
    # Get test dataloader
    test_loader = data_module.test_dataloader()
    
    # Get a batch of samples
    batch = next(iter(test_loader))
    images, labels = batch
    
    # Move images to the same device as the model
    device = next(model.parameters()).device
    images = images.to(device)
    
    # Limit to num_samples
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    # Create visualizer
    visualizer = AttentionVisualizer()
    
    # Create save directory if specified
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize attention for each sample
    for i, (image, label) in enumerate(zip(images, labels)):
        # Add batch dimension
        image = image.unsqueeze(0)
        
        # Get model prediction
        with torch.no_grad():
            logits, _, _, _ = model(image)
            # TODO: check if this is correct
            pred = logits[-1].argmax(dim=1).item()
        
        # Create visualization
        if save_dir:
            name = f"attention_sample_{i}_true_{label.item()}_pred_{pred}.png"
            save_path = save_dir 
            if label.item() != pred:
                save_path = save_dir / "errors" 
            save_path.mkdir(parents=True, exist_ok=True)
            save_path = save_path / name
        else:
            save_path = None
            
        visualizer.visualize_attention_sequence(
            model,
            image,
            save_path=str(save_path) if save_path else None,
            glimpse_size=model.hparams.glimpse_size,
            num_patches=model.hparams.num_patches
        )
        
        print(f"Sample {i}: True label = {label.item()}, Predicted = {pred}")
        if save_path:
            print(f"Visualization saved to {save_path}")


def main(checkpoint_path: str, num_samples: int = 10):
    """Main function to load model and create visualizations."""
    # Add the project root to the Python path
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))
    
    # Load the training configuration
    config = load_config(config_name='train')
    print("Configuration loaded")
    
    # Load the model from checkpoint
    model = RecurrentAttentionModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    print(f"Model loaded from {checkpoint_path}")
    
    # Initialize the data module
    data_module = hydra.utils.instantiate(config.datamodule)
    data_module.setup()
    print("Data module initialized")
    
    # Get the checkpoint directory for saving visualizations
    checkpoint_dir = Path(checkpoint_path).parent.parent / "diags"
    
    # Create visualizations
    visualize_test_samples(
        model=model,
        data_module=data_module,
        num_samples=num_samples,
        save_dir=checkpoint_dir / "attention_visualizations"
    )
    print("Visualization completed!")


if __name__ == "__main__":
    ckpt_path = "/mnt/c/users/victor/documents/projects/Glimpse Attention Networks/results/example_experiment/2025-06-02_22-49-59/metrics/version_0/checkpoints/ram-epoch=15-val_accuracy=0.867.ckpt"
    num_samples = 20
    main(ckpt_path, num_samples) 