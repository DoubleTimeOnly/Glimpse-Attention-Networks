import torch
from torchvision import datasets, transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader

class SVHNDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/svhn",
        batch_size: int = 32,
        num_workers: int = 4,
        train_val_split: float = 0.9,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
        ])
        
    def prepare_data(self):
        # Download the dataset if it doesn't exist
        datasets.SVHN(
            root=self.data_dir,
            split='train',
            download=True,
            transform=None
        )
        datasets.SVHN(
            root=self.data_dir,
            split='test',
            download=True,
            transform=None
        )
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            # Load full training set
            full_train = datasets.SVHN(
                root=self.data_dir,
                split='train',
                download=False,
                transform=self.transform
            )
            
            # Split into train and validation
            train_size = int(len(full_train) * self.train_val_split)
            val_size = len(full_train) - train_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                full_train, [train_size, val_size]
            )
            
        if stage == 'test' or stage is None:
            self.test_dataset = datasets.SVHN(
                root=self.data_dir,
                split='test',
                download=False,
                transform=self.transform
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        ) 