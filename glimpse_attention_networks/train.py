"""
Training Script for Recurrent Attention Model (RAM)
This script demonstrates how to train the RAM model on SVHN dataset
"""
from pathlib import Path
from datetime import datetime
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from glimpse_attention_networks.training.initializers import initialize_callbacks, initialize_loggers
from glimpse_attention_networks.utils.config import load_config


def train_ram_model(config: DictConfig, experiment_name: str):
    """Main training function for RAM model."""
    
    # Initialize model
    model = hydra.utils.instantiate(config.model)
    
    # Initialize callbacks
    callbacks = initialize_callbacks(config.callbacks)
    
    # Assemble full log directory path with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path(config.log_dir) / experiment_name / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize loggers
    loggers = initialize_loggers(log_dir=str(log_dir))
    
    # Initialize trainer with callbacks and loggers
    trainer = hydra.utils.instantiate(
        config.trainer, 
        callbacks=callbacks,
        logger=loggers
    )

    # Initialize data module
    data_module = hydra.utils.instantiate(config.datamodule)

    # Train the model
    trainer.fit(model, data_module)
    
    # Test the model
    trainer.test(ckpt_path="best", datamodule=data_module)
    print(f"Test results saved to {log_dir}")


if __name__ == "__main__":
    # Train the model
    config = load_config(
        config_name='train', 
        overrides=[]
    )
    print("Starting RAM training...")
    experiment_name = "example_experiment"
    train_ram_model(config, experiment_name)
    print("Training completed!")