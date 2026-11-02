# src/api/main.py
import os
import time

import mlflow
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import XLMRobertaTokenizerFast

app = FastAPI(
    title="Toxic Comment Classifier API",
    description="SageMaker-compatible inference container for abusive language detection.",
    version="1.0.0"
)

class InferenceRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, description="Text to classify")

class InferenceResponse(BaseModel):
    is_toxic: bool
    confidence: float
    latency_ms: float

MODEL = None
TOKENIZER = None

@app.on_event("startup")
def load_model():
    """Loads the MLflow artifact and Tokenizer into memory when the container starts."""
    global MODEL, TOKENIZER
    sm_model_dir = os.getenv("SM_MODEL_DIR") 
    
    try:
        if sm_model_dir:
            print(f"SageMaker env detected. Loading model from {sm_model_dir}...")
            MODEL = mlflow.pytorch.load_model(sm_model_dir) # type: ignore
        else:
            print("Local env detected. Resolving latest MLflow run...")
            mlflow.set_tracking_uri("sqlite:///mlruns.db")
            
            experiment = mlflow.get_experiment_by_name("Multilingual_Toxicity_Detection")
            assert experiment is not None
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1
            )
            
            latest_run_id = runs.iloc[0].run_id
            model_uri = f"runs:/{latest_run_id}/model_artifacts"
            
            print(f"Loading model dynamically from {model_uri}...")
            MODEL = mlflow.pytorch.load_model(model_uri) # type: ignore
            
        MODEL.eval()
        
        # Initialize Tokenizer globally to prevent reloading on every POST request
        print("Loading Tokenizer...")
        TOKENIZER = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")
        print("✅ Model and Tokenizer loaded successfully!")
    except Exception as e:  # noqa: BLE001
        print(f"❌ Failed to load model: {e}")

@app.get("/ping")
def ping():
    if MODEL is not None:
        return {"status": "Healthy"}
    raise HTTPException(status_code=503, detail="Model not loaded")

@app.post("/invocations", response_model=InferenceResponse)
def predict(request: InferenceRequest):
    start_time = time.time()
    
    if MODEL is None or TOKENIZER is None:
        raise HTTPException(status_code=503, detail="Model/Tokenizer not loaded")
        
    # Preprocessing
    inputs = TOKENIZER(
        request.text, 
        return_tensors="pt", 
        truncation=True, 
        padding="max_length", 
        max_length=128
    )
    
    device = next(MODEL.parameters()).device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Inference
    with torch.no_grad():
        outputs = MODEL(input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(outputs.logits).squeeze().item()
    
    latency = (time.time() - start_time) * 1000
    
    return InferenceResponse(
        is_toxic=bool(probs >= 0.5),
        confidence=probs,
        latency_ms=latency
    )