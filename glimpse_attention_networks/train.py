"""
Training Script for Recurrent Attention Model (RAM)
This script demonstrates how to train the RAM model on SVHN dataset
"""
from pathlib import Path    

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from models.ram_lit_module import RecurrentAttentionModel
from datasets.svhn_datamodule import SVHNDataModule

def train_ram_model():
    """Main training function for RAM model."""
    
    # Hyperparameters
    config = {
        'glimpse_size': 8,
        'num_patches': 3,
        'num_glimpses': 6,
        'num_classes': 10,
        'channels': 3,  # SVHN has 3 channels (RGB)
        'hidden_size': 256,
        'glimpse_hidden': 256,
        'location_std': 0.17,
        'learning_rate': 1e-3,
        'baseline_coeff': 0.5,
        'batch_size': 128,
        # 'max_epochs': 50,
        'max_epochs': 5,
        'image_size': 32,  # SVHN images are 32x32
        'num_workers': 4
    }
    
    # Initialize model
    model = RecurrentAttentionModel(**{k: v for k, v in config.items() 
                                     if k in ['glimpse_size', 'num_patches', 'num_glimpses', 
                                             'num_classes', 'channels', 'hidden_size', 
                                             'glimpse_hidden', 'location_std', 'learning_rate', 
                                             'baseline_coeff']})
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_accuracy',
        mode='max',
        save_top_k=1,
        filename='ram-{epoch:02d}-{val_accuracy:.3f}'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_accuracy',
        mode='max',
        patience=10,
        verbose=True
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator='auto',
        devices='auto',
        # precision=16,  # Mixed precision training
        precision=32,
        log_every_n_steps=50,
        val_check_interval=0.5,  # Validate twice per epoch
    )

    # Create data module for SVHN
    data_module = SVHNDataModule(
        data_dir='/mnt/c/users/victor/documents/projects/Glimpse Attention Networks/data/svhn',
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )

    # Train the model
    trainer.fit(model, data_module)
    
    # Test the best model
    best_model = RecurrentAttentionModel.load_from_checkpoint(
        checkpoint_callback.best_model_path
    )
    
    # Evaluation on test set
    trainer.test(best_model, data_module)
    
    return best_model, trainer, data_module




if __name__ == "__main__":
    # Train the model
    print("Starting RAM training...")
    model, trainer, data_module = train_ram_model()
    print("Training completed!")
    
    # Example of attention visualization
    print("\nCreating attention visualization example...")
    
    # Get one sample from validation set
    sample_batch = next(iter(data_module.val_dataloader()))
    sample_image, sample_label = sample_batch
    
    # Visualize attention
    visualizer = AttentionVisualizer()
    visualizer.visualize_attention_sequence(model, sample_image, save_path='attention_visualization.png')