"""
Configuration utilities for loading and managing Hydra configurations.
"""
from typing import List, Optional, Union
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

CONFIG_DIR = Path(__file__).parent.parent.parent / 'configs'

def load_config(
    config_name: str,
    overrides: List[str] = [],
    version_base: Optional[str] = None,
    config_dir: Union[str, Path] = CONFIG_DIR,
) -> DictConfig:
    with initialize_config_dir(
        config_dir=str(config_dir),
        version_base=version_base,
    ):
        # Compose the configuration
        cfg = compose(
            config_name=config_name,
            overrides=overrides,
        )
        
    return cfg 