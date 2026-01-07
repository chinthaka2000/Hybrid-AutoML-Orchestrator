"""
Tests for policy engine logic.
"""

import unittest
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mcp.policy_engine import PolicyEngine

class TestPolicyEngine(unittest.TestCase):
    def test_retraining_policy(self):
        config = {'policies': {'retraining': {'threshold': 0.5}}}
        engine = PolicyEngine(config)
        
        # Case 1: Drift > Threshold
        context = {'action': 'retrain', 'metrics': {'drift_score': 0.6}}
        self.assertTrue(engine.check_policy(context))
        
        # Case 2: Drift <= Threshold
        context = {'action': 'retrain', 'metrics': {'drift_score': 0.4}}
        self.assertFalse(engine.check_policy(context))
        
    def test_rollback_policy(self):
        config = {'policies': {'rollback': {'threshold': 0.1}}}
        engine = PolicyEngine(config)
        
        # Case 1: Drop > Threshold
        context = {'action': 'rollback', 'metrics': {'performance_drop': 0.15}}
        self.assertTrue(engine.check_policy(context))

if __name__ == '__main__':
    unittest.main()
