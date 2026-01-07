"""
Configuration loader.
"""

import yaml
from pathlib import Path

class ConfigLoader:
    def __init__(self, config_dir: str):
        self.config_dir = Path(config_dir)

    def load_config(self, config_name: str) -> dict:
        """
        Loads a yaml config file.
        """
        config_path = self.config_dir / f"{config_name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file {config_path} not found.")
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
