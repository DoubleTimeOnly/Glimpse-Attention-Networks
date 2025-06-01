"""
Training Script for Recurrent Attention Model (RAM)
This script demonstrates how to train the RAM model on SVHN dataset
"""
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from glimpse_attention_networks.training.initializers import initialize_callbacks
from glimpse_attention_networks.utils.config import load_config


def train_ram_model(config: DictConfig):
    """Main training function for RAM model."""
    
    # Initialize model
    model = hydra.utils.instantiate(config.model)
    
    # Initialize callbacks
    callbacks = initialize_callbacks(config.callbacks)
    
    # Initialize trainer with callbacks
    trainer = hydra.utils.instantiate(
        config.trainer, 
        callbacks=callbacks
    )

    # Initialize data module
    data_module = hydra.utils.instantiate(config.datamodule)

    # Train the model
    trainer.fit(model, data_module)
    
    # Test the model
    trainer.test(ckpt_path="best", datamodule=data_module)


if __name__ == "__main__":
    # Train the model
    config = load_config(
        config_name='train', 
        overrides=[]
    )
    print("Starting RAM training...")
    train_ram_model(config)
    print("Training completed!")