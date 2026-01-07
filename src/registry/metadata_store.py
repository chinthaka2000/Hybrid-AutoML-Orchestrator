"""
Stores model metadata and configurations.
"""

import json
from pathlib import Path

class MetadataStore:
    def __init__(self, key: str, storage_dir: str = "models/metadata"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.key = key

    def save_metadata(self, id: str, metadata: dict):
        file_path = self.storage_dir / f"{id}.json"
        with open(file_path, 'w') as f:
            json.dump(metadata, f, indent=4)

    def load_metadata(self, id: str) -> dict:
        file_path = self.storage_dir / f"{id}.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        return {}
