"""
Interface with H2O AutoML.
"""

import h2o
from h2o.automl import H2OAutoML

class H2OClient:
    def __init__(self, url="http://localhost:54321"):
        self.url = url
        try:
            h2o.init(url=self.url)
        except Exception as e:
            print(f"Warning: Could not connect to H2O at {url}: {e}")

    def train_automl(self, train_data_path, target, max_models=20, max_runtime_secs=3600, project_name="automl_experiment"):
        """
        Trains H2O AutoML on the provided data.
        """
        try:
            data = h2o.import_file(train_data_path)
            
            # Identify predictors and response
            x = data.columns
            y = target
            if y in x:
                x.remove(y)
            
            # For classification, ensure target is a factor
            # data[y] = data[y].asfactor() # basic assumption handling
            
            aml = H2OAutoML(max_models=max_models, 
                            max_runtime_secs=max_runtime_secs, 
                            project_name=project_name,
                            seed=1234)
            
            aml.train(x=x, y=y, training_frame=data)
            return aml
        except Exception as e:
            print(f"AutoML training failed: {e}")
            raise e
