"""
Logic to trigger retraining based on conditions.
"""

from mcp.policy_engine import PolicyEngine

class RetrainingTrigger:
    def __init__(self, policy_engine: PolicyEngine):
        self.policy_engine = policy_engine

    def should_retrain(self, metrics: dict) -> bool:
        """
        Determines if retraining is needed based on policies.
        """
        context = {
            'action': 'retrain',
            'metrics': metrics
        }
        # PolicyEngine returns True if policy is VIOLATED (e.g. drift > threshold)
        # Wait, check_retraining_policy returns True if drift > threshold.
        # So if True, we SHOULD retrain.
        return self.policy_engine.check_policy(context)
