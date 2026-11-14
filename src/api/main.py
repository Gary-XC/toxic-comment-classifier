# src/api/main.py
import time
from contextlib import asynccontextmanager

import numpy as np
import tritonclient.grpc as grpcclient
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import XLMRobertaTokenizerFast

TOKENIZER = None
TRITON_CLIENT = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global TOKENIZER, TRITON_CLIENT
    print("Loading Tokenizer...")
    TOKENIZER = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")
    
    print("Connecting to Triton Inference Server...")
    try:
        TRITON_CLIENT = grpcclient.InferenceServerClient(url="localhost:8001")
        if not TRITON_CLIENT.is_server_ready():
            raise Exception("Triton server is not ready")  # noqa: TRY002
        print("✅ Connected to Triton!")
    except Exception as e:  # noqa: BLE001
        print(f"❌ Triton Connection Failed: {e}")
    
    yield

app = FastAPI(title="Triton AI Gateway", lifespan=lifespan)

class InferenceRequest(BaseModel):
    text: str

@app.post("/invocations")
def predict(request: InferenceRequest):
    start_time = time.time()
    
    # tokenize (CPU)
    inputs = TOKENIZER(request.text, truncation=True, padding="max_length", max_length=128) # type: ignore
    input_ids = np.array([inputs["input_ids"]], dtype=np.int64)
    attention_mask = np.array([inputs["attention_mask"]], dtype=np.int64)
    
    # package for the Triton gRPC
    triton_inputs = [
        grpcclient.InferInput("INPUT__0", input_ids.shape, "INT64"),
        grpcclient.InferInput("INPUT__1", attention_mask.shape, "INT64")
    ]
    triton_inputs[0].set_data_from_numpy(input_ids)
    triton_inputs[1].set_data_from_numpy(attention_mask)
    
    # requesting GPU Inference from Triton
    result = TRITON_CLIENT.infer(model_name="toxic-comment-xlmr", inputs=triton_inputs) # type: ignore
    
    # process Output
    logits_array = result.as_numpy("OUTPUT__0") # type: ignore
    
    # extract the single logit value from the nested array structure (like [[-2.4]])
    # use .item() to safely convert a 1-element numpy array into a native Python float
    logit = logits_array.item()  # type: ignore
    
    # Applying sigmoid to convert the raw logit into a probability (0.0 to 1.0)
    toxic_prob = 1 / (1 + np.exp(-logit))
    
    latency = (time.time() - start_time) * 1000
    
    return {
        "is_toxic": bool(toxic_prob >= 0.5),
        "confidence": toxic_prob,
        "latency_ms": latency
    }