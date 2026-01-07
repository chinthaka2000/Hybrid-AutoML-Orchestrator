"""
Data drift detection logic.
"""

import pandas as pd
from scipy.stats import ks_2samp
import os

class DataDriftDetector:
    def __init__(self, reference_data_path):
        self.reference_data_path = reference_data_path
        self.reference_data = None
        if os.path.exists(reference_data_path):
            self.reference_data = pd.read_csv(reference_data_path)

    def detect_drift(self, new_data_path, threshold=0.05) -> bool:
        """
        Detects drift between new data and reference data using KS test for numerical columns.
        Returns True if drift is detected in ANY column.
        """
        if self.reference_data is None:
            print("Reference data not found. proper drift detection skipped.")
            return False
            
        try:
            new_data = pd.read_csv(new_data_path)
            # Check numerical columns
            drift_detected = False
            for col in new_data.select_dtypes(include=['number']).columns:
                 if col in self.reference_data.columns:
                     stat, p_value = ks_2samp(self.reference_data[col], new_data[col])
                     if p_value < threshold:
                         print(f"Drift detected in column {col} (p-value: {p_value})")
                         drift_detected = True
            
            return drift_detected
        except Exception as e:
            print(f"Error during drift detection: {e}")
            return False
