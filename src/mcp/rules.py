"""
Predefined hard rules for system control.
"""

def check_hard_rules(metrics: dict, config: dict) -> bool:
    """
    Returns False if any hard rule is violated.
    Expects 'training_time_hours' and 'cost_usd' in metrics.
    """
    budget = config.get('policies', {}).get('budget', {})
    max_time = budget.get('max_training_time_hours', 4)
    max_cost = budget.get('max_cost_usd', 50)
    
    if metrics.get('training_time_hours', 0) > max_time:
        return False
        
    if metrics.get('cost_usd', 0) > max_cost:
        return False
        
    return True
