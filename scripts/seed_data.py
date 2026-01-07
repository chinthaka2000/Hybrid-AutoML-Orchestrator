"""
Seeds data for testing or experimentation.
"""

import pandas as pd
import numpy as np
import os

def seed_data():
    """
    Creates dummy data for testing:
    - data/reference/reference.csv (Normal distribution)
    - data/raw/current.csv (Shifted distribution - Drift)
    """
    os.makedirs("data/reference", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)
    
    # Reference data
    print("Generating reference data...")
    ref_df = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, 100),
        'feature_2': np.random.normal(10, 2, 100),
        'target': np.random.choice([0, 1], 100)
    })
    ref_df.to_csv("data/reference/reference.csv", index=False)
    
    # Current data (Drifted)
    print("Generating drifted data...")
    curr_df = pd.DataFrame({
        'feature_1': np.random.normal(2, 1, 100), # Mean shift 0 -> 2
        'feature_2': np.random.normal(10, 2, 100),
        'target': np.random.choice([0, 1], 100)
    })
    curr_df.to_csv("data/raw/current.csv", index=False)
    
    print("Data seeded successfully.")

if __name__ == "__main__":
    seed_data()
