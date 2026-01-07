from .model_registry import ModelRegistry

class VersionManager:
    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    def get_latest_version(self, status: str = "production") -> dict:
        """
        Returns the latest model with the given status.
        """
        registry_data = self.registry._load_registry()
        candidates = [
            m for m in registry_data.values() 
            if m.get('status') == status
        ]
        
        if not candidates:
            return None
            
        # Sort by registration timestamp
        candidates.sort(key=lambda x: x.get('registered_at', ''), reverse=True)
        return candidates[0]
