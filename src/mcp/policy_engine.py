"""
Core policy enforcement logic.
"""

class PolicyEngine:
    def __init__(self, config):
        self.config = config

    def check_policy(self, context: dict) -> bool:
        """
        Evaluates policies against a context.
        Context expected keys: 'action', 'metrics', 'current_state'
        """
        action = context.get('action')
        metrics = context.get('metrics', {})
        print(f"DEBUG: Checking policy for action: {action}")
        
        if action == 'retrain':
            return self._check_retraining_policy(metrics)
        elif action == 'rollback':
            return self._check_rollback_policy(metrics)
        
        return True

    def _check_retraining_policy(self, metrics: dict) -> bool:
        policy = self.config.get('policies', {}).get('retraining', {})
        threshold = policy.get('threshold', 0.05)
        drift_score = metrics.get('drift_score', 0.0)
        
        print(f"DEBUG: Drift Score: {drift_score}, Threshold: {threshold}, Policy: {policy}")
        
        if drift_score > threshold:
            return True
        return False

    def _check_rollback_policy(self, metrics: dict) -> bool:
        policy = self.config.get('policies', {}).get('rollback', {})
        threshold = policy.get('threshold', 0.10)
        performance_drop = metrics.get('performance_drop', 0.0)
        
        if performance_drop > threshold:
            return True
        return False
