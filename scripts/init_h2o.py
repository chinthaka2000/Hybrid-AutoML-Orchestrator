"""
Initializes H2O AutoML.
"""

import h2o

def init_h2o():
    try:
        h2o.init()
        print("H2O initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize H2O: {e}")

if __name__ == "__main__":
    init_h2o()
