"""
Exposes models for inference (API or batch).
"""

import h2o
import pandas as pd

class InferenceService:
    def __init__(self):
        # Ensure H2O is connected
        try:
            h2o.init(url="http://localhost:54321")
        except:
            pass

    def predict(self, model_path: str, input_data):
        """
        Loads a model and generates predictions.
        input_data: pandas DataFrame or list of lists
        """
        try:
            model = h2o.load_model(model_path)
            
            # Convert to H2OFrame
            if isinstance(input_data, pd.DataFrame):
                hf = h2o.H2OFrame(input_data)
            else:
                hf = h2o.H2OFrame(python_obj=input_data)
                
            preds = model.predict(hf)
            return preds.as_data_frame()
        except Exception as e:
            print(f"Inference failed: {e}")
            raise e
