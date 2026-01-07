"""
Main entry point for the Hybrid AutoML Orchestrator.
"""

import sys
import os

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.logger import setup_logger
from utils.config_loader import ConfigLoader

logger = setup_logger(__name__)

def main():
    """
    Main execution loop.
    """
    logger.info("Starting Hybrid AutoML Orchestrator...")
    
    # 1. Load Configs
    config_loader = ConfigLoader("configs")
    try:
        mcp_config = config_loader.load_config("mcp_policy")
        automl_config = config_loader.load_config("automl_config")
    except Exception as e:
        logger.error(f"Failed to load configs: {e}")
        return

    # 2. Initialize Components
    from mcp.policy_engine import PolicyEngine
    from monitoring.data_drift import DataDriftDetector
    from orchestrator.retraining_trigger import RetrainingTrigger
    from orchestrator.automl_runner import AutoMLRunner
    from registry.model_registry import ModelRegistry
    from deployment.inference_service import InferenceService

    policy_engine = PolicyEngine(config={'policies': mcp_config.get('policies', {})})
    drift_detector = DataDriftDetector("data/reference/reference.csv")
    trigger = RetrainingTrigger(policy_engine)
    registry = ModelRegistry()
    runner = AutoMLRunner(automl_config.get('automl', {}), registry)
    
    # 3. Monitoring Step
    logger.info("Checking for data drift...")
    new_data_path = "data/raw/current.csv"
    if not os.path.exists(new_data_path):
        logger.warning(f"No new data found at {new_data_path}. Skipping.")
        return

    is_drift = drift_detector.detect_drift(new_data_path)
    logger.info(f"Drift Detected: {is_drift}")
    
    # 4. Policy Check
    should_train = trigger.should_retrain({'drift_score': 1.0 if is_drift else 0.0}) # Simplified metric passing
    
    if should_train:
        logger.info("Policy triggered: RETRAINING started.")
        try:
            model_id = runner.run_experiment(new_data_path, target="target")
            logger.info(f"Retraining completed. New model: {model_id}")
            
            # 5. Mock Inference Check
            logger.info("Verifying model with inference...")
            service = InferenceService()
            model_data = registry.get_model(model_id)
            if model_data:
                preds = service.predict(model_data['path'], [[1.0, 10.0]]) # Dummy input
                logger.info(f"Prediction sample:\n{preds}")
                
        except Exception as e:
            logger.error(f"Retraining failed: {e}")
    else:
        logger.info("No retraining needed according to policy.")
    
    logger.info("Orchestrator finished.")

if __name__ == "__main__":
    main()
