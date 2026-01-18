import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import h2o

class ModelEvaluator:
    def __init__(self):
        pass

    def evaluate(self, model, test_data_path: str, target: str) -> dict:
        """
        Evaluates the model and returns a dictionary of metrics.
        """
        try:
            # Load test data
            if isinstance(test_data_path, str):
                hf = h2o.import_file(test_data_path)
            else:
                hf = test_data_path # Assume it's already an H2OFrame

            # Predict
            preds = model.predict(hf).as_data_frame()
            actuals = hf[target].as_data_frame()
            
            # SKLearn expects arrays
            y_true = actuals.iloc[:, 0].astype(str).tolist()
            y_pred = preds.iloc[:, 0].astype(str).tolist()

            # Calculate metrics
            report = classification_report(y_true, y_pred, output_dict=True)
            cm = confusion_matrix(y_true, y_pred).tolist()

            evaluation = {
                "classification_report": report,
                "confusion_matrix": cm,
                "model_id": model.model_id,
                "details": {
                    "accuracy": report.get("accuracy", 0.0),
                    "macro_avg_f1": report.get("macro avg", {}).get("f1-score", 0.0)
                }
            }
            return evaluation

        except Exception as e:
            print(f"Evaluation failed: {e}")
            return {}
