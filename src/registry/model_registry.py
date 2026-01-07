import json
import os
import shutil
from datetime import datetime
from pathlib import Path

class ModelRegistry:
    def __init__(self, registry_path: str = "models/registry.json"):
        self.registry_path = Path(registry_path)
        if not self.registry_path.exists():
            self._save_registry({})

    def _load_registry(self) -> dict:
        with open(self.registry_path, 'r') as f:
            return json.load(f)

    def _save_registry(self, data: dict):
        with open(self.registry_path, 'w') as f:
            json.dump(data, f, indent=4)

    def register_model(self, model_path: str, metadata: dict) -> str:
        """
        Registers a new model. Returns model ID.
        """
        registry = self._load_registry()
        model_id = f"model_{int(datetime.now().timestamp())}"
        
        # Ensure metadata has timestamp and id
        metadata['id'] = model_id
        metadata['registered_at'] = datetime.now().isoformat()
        metadata['path'] = model_path
        metadata['status'] = 'staging'
        
        registry[model_id] = metadata
        self._save_registry(registry)
        return model_id

    def get_model(self, model_id: str) -> dict:
        registry = self._load_registry()
        return registry.get(model_id)

    def update_status(self, model_id: str, status: str):
        registry = self._load_registry()
        if model_id in registry:
            registry[model_id]['status'] = status
            self._save_registry(registry)
        else:
            raise ValueError(f"Model {model_id} not found.")
