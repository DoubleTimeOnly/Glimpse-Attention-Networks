"""
Training callback initializers for the Recurrent Attention Model.
"""
from typing import List, Dict, Any
from pathlib import Path

import hydra
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import CSVLogger


def initialize_callbacks(callbacks_config: Dict[str, Any]) -> List[Callback]:
    callbacks = []
    
    for callback_name, callback_config in callbacks_config.items():
        # Initialize callback using Hydra's instantiate
        callback = hydra.utils.instantiate(callback_config)
        callbacks.append(callback)
        
    return callbacks


def initialize_loggers(log_dir: str) -> List[CSVLogger]:
    csv_logger = CSVLogger(
        save_dir=log_dir,
        name="metrics",
        version=None  # No versioning, just use the experiment name
    )
    
    return [csv_logger] 