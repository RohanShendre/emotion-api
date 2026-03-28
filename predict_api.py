"""
FastAPI prediction server for the emotion BERT model.

Usage:
    pip install fastapi uvicorn transformers torch numpy pydantic
    python -m uvicorn predict_api:app --host 0.0.0.0 --port 8000

POST /predict
Body: {"text": "I am happy"}  OR {"texts": ["I am happy", "I am sad"]}
Response: {"predictions": [ {"label": "...", "scores": [{"label": "...","score":0.82}, ...], "top_label": "..."} , ... ] }
python -m uvicorn predict_api:app --host 0.0.0.0 --port 8000
"""

from typing import List, Optional
import os
import logging

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------
# Config
# -----------------------
MODEL_DIR = os.getenv("EMOTION_MODEL_DIR", "ShaileshWakpaijan/emotion-model")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_BATCH = 64
MAX_INPUT_LENGTH = 512

# -----------------------
# Logging
# -----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("predict_api")

# -----------------------
# App
# -----------------------
app = FastAPI(title="Emotion Prediction API")

# -----------------------
# Request/Response Models
# -----------------------
class PredictRequest(BaseModel):
    text: Optional[str] = None
    texts: Optional[List[str]] = None
    top_k: Optional[int] = None

class LabelScore(BaseModel):
    label: str
    score: float

class Prediction(BaseModel):
    label: str
    scores: List[LabelScore]
    top_label: str

class PredictResponse(BaseModel):
    predictions: List[Prediction]

# -----------------------
# Load model
# -----------------------
def load_labels(model_dir):
    path = os.path.join(model_dir, "label_map.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip().split("\t")[-1] for line in f if line.strip()]
    return None

@app.on_event("startup")
def load_model():
    global tokenizer, model, labels

    if not os.path.isdir(MODEL_DIR):
        raise RuntimeError(f"Model directory not found: {MODEL_DIR}")

    logger.info("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
    model.eval()

    labels = load_labels(MODEL_DIR)
    if labels is None:
        labels = [model.config.id2label[i] for i in range(len(model.config.id2label))]

    logger.info(f"Loaded labels: {labels}")

# -----------------------
# Prediction function
# -----------------------
def predict_batch(texts: List[str]):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    return probs

# -----------------------
# Health endpoint
# -----------------------
@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "labels": len(labels)}

# -----------------------
# Predict endpoint
# -----------------------
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if req.text is not None:
        text = req.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="field 'text' is empty")
        texts = [text]

    elif req.texts is not None:
        if not isinstance(req.texts, list) or len(req.texts) == 0:
            raise HTTPException(status_code=400, detail="field 'texts' must be a non-empty list of strings")

        texts = [t.strip() for t in req.texts]
        if any(not t for t in texts):
            raise HTTPException(status_code=400, detail="empty string found in 'texts' list")

    else:
        raise HTTPException(status_code=400, detail="Provide 'text' or 'texts'")

    if len(texts) > MAX_BATCH:
        raise HTTPException(status_code=400, detail=f"Max batch size is {MAX_BATCH}")

    probs_list = predict_batch(texts)

    predictions = []
    for probs in probs_list:
        scores = [
            {"label": labels[i], "score": float(round(probs[i], 6))}
            for i in range(len(labels))
        ]

        top_label = max(scores, key=lambda x: x["score"])["label"]

        if req.top_k is not None:
            if req.top_k < 1:
                raise HTTPException(status_code=400, detail="top_k must be >= 1")
            scores = sorted(scores, key=lambda x: x["score"], reverse=True)[:req.top_k]
            top_label = scores[0]["label"]

        predictions.append({
            "label": top_label,
            "scores": scores,
            "top_label": top_label
        })

    return {"predictions": predictions}

# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("predict_api:app", host="0.0.0.0", port=port)