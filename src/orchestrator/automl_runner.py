"""
Runs AutoML experiments and model selection.
"""

import h2o
from automl.h2o_client import H2OClient
from registry.model_registry import ModelRegistry
import os

class AutoMLRunner:
    def __init__(self, config: dict, registry: ModelRegistry):
        self.config = config
        self.registry = registry
        self.h2o_client = H2OClient()

    def run_experiment(self, data_path: str, target: str):
        """
        Runs an AutoML experiment and registers the best model.
        """
        print(f"Starting AutoML experiment on {data_path}...")
        
        # Train
        aml = self.h2o_client.train_automl(
            data_path, 
            target,
            max_models=self.config.get('max_models', 10),
            max_runtime_secs=self.config.get('max_runtime_secs', 3600)
        )
        
        # Get best model
        best_model = aml.leader
        print(f"Best model found: {best_model.model_id}")
        
        # Save model
        models_dir = os.path.abspath("models/staging")
        os.makedirs(models_dir, exist_ok=True)
        saved_path = h2o.save_model(model=best_model, path=models_dir, force=True)
        
        # Register
        metrics = {
            "algo": best_model.algo,
            "rmse": best_model.rmse(),
            "auc": best_model.auc() if best_model.algo != "StackedEnsemble" else None # StackedEnsemble might not have AUC in some cases
        }
        model_id = self.registry.register_model(saved_path, metrics)
        print(f"Model registered with ID: {model_id}")
        
        return model_id
