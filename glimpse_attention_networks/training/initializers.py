"""
Training callback initializers for the Recurrent Attention Model.
"""
from typing import List, Dict, Any

import hydra
from pytorch_lightning.callbacks import Callback


def initialize_callbacks(callbacks_config: Dict[str, Any]) -> List[Callback]:
    callbacks = []
    
    for callback_name, callback_config in callbacks_config.items():
        # Initialize callback using Hydra's instantiate
        callback = hydra.utils.instantiate(callback_config)
        callbacks.append(callback)
        
    return callbacks 