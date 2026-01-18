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
    from mcp.decision_engine import DecisionEngine
    from monitoring.data_drift import DataDriftDetector
    from orchestrator.retraining_trigger import RetrainingTrigger
    from orchestrator.automl_runner import AutoMLRunner
    from registry.model_registry import ModelRegistry
    from automl.model_evaluator import ModelEvaluator

    policy_engine = PolicyEngine(config={'policies': mcp_config.get('policies', {})})
    decision_engine = DecisionEngine() # Pulls API key from env
    drift_detector = DataDriftDetector("data/reference/reference.csv")
    trigger = RetrainingTrigger(policy_engine)
    registry = ModelRegistry()
    runner = AutoMLRunner(automl_config.get('automl', {}), registry)
    evaluator = ModelEvaluator()
    
    # 3. Monitoring Step
    logger.info("Checking for data drift...")
    new_data_path = "data/raw/current.csv"
    if not os.path.exists(new_data_path):
        logger.warning(f"No new data found at {new_data_path}. Skipping.")
        return

    is_drift = drift_detector.detect_drift(new_data_path)
    logger.info(f"Drift Detected: {is_drift}")
    
    # 4. Cognitive Reasoning Check
    latest_model_info = registry.get_model("TODO_LATEST_ID") # In real implementation, query version manager
    
    context = {
        'drift_status': 'Detected' if is_drift else 'None',
        'current_metrics': {'accuracy': 0.85}, # Placeholder
        'policies': mcp_config.get('policies', {})
    }
    
    decision = decision_engine.get_recommendation(context)
    logger.info(f"LLM Recommendation: {decision}")
    
    should_train = False
    if decision.get("action") == "RETRAINING" or (is_drift and trigger.should_retrain({'drift_score': 1.0})):
         should_train = True
    
    if should_train:
        logger.info("Policy/LLM triggered: RETRAINING started.")
        try:
            model_id = runner.run_experiment(new_data_path, target="target")
            logger.info(f"Retraining completed. New model: {model_id}")
            
            # 5. Advanced Evaluation
            # Need test data split in real world
            evaluation = evaluator.evaluate(runner.h2o_client.train_automl(new_data_path, "target").leader, new_data_path, "target")
            logger.info(f"New Model Evaluation: {evaluation.get('details')}")
                
        except Exception as e:
            logger.error(f"Retraining failed: {e}")
    else:
        logger.info("No retraining needed according to policy/LLM.")
    
    logger.info("Orchestrator finished.")

if __name__ == "__main__":
    main()
