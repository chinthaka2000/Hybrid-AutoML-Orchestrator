from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import h2o
import pandas as pd
import os
from contextlib import asynccontextmanager

# Global model
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    global model
    try:
        h2o.init(url=os.getenv("H2O_URL", "http://localhost:54321"))
        model_path = os.getenv("MODEL_PATH") # e.g., models/production/model_id
        if model_path:
            model = h2o.load_model(model_path)
            print(f"Loaded model from {model_path}")
        else:
            print("Warning: MODEL_PATH env var not set.")
    except Exception as e:
        print(f"Startup failed: {e}")
    yield
    # Shutdown logic if needed

app = FastAPI(lifespan=lifespan)

class PredictionRequest(BaseModel):
    data: list

@app.post("/predict")
async def predict(request: PredictionRequest):
    global model
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # data is expected to be list of lists or dict
        hf = h2o.H2OFrame(python_obj=request.data)
        preds = model.predict(hf).as_data_frame()
        return {"predictions": preds.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
