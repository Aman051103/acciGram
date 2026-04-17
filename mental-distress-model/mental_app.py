"""
Mental Health Distress Detection Microservice
===============================================
POST /classify
  Input  : [{"username": "user1", "text": "I feel empty..."}, ...]
  Output : ["user1", "user3"]   -- usernames needing attention

POST /classify/detailed
  Input  : same
  Output : [{"username": "user1", "prediction": "Depression",
             "confidence": 0.87, "top_2": [...], "needs_attention": true}]

GET  /health
  Output : {"status": "ok", "model": "...", "classes": [...]}
"""

import re
import json
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import uvicorn

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_DIR  = "roberta_mental_health_model"   # unzip roberta_mental_health_model.zip here
MAX_LEN    = 128
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Classes that should trigger an alert — adjust to your use case
ALERT_CLASSES = {"Suicidal", "Depression", "Anxiety", "Bipolar",
                 "Stress", "Personality disorder"}

# ── Load model at startup ─────────────────────────────────────────────────────
print(f"Loading model from '{MODEL_DIR}' on {DEVICE}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(DEVICE).eval()

# Load label mappings saved during training
with open(f"{MODEL_DIR}/id2label.json") as f:
    ID2LABEL = {int(k): v for k, v in json.load(f).items()}
with open(f"{MODEL_DIR}/label2id.json") as f:
    LABEL2ID = json.load(f)

print(f"✅ Model ready | Classes: {list(ID2LABEL.values())}")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Mental Health Distress Detection Service",
    description="Classifies mental health distress from text using RoBERTa.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────────────────────
class UserMessage(BaseModel):
    username: str
    text: str

class DetailedResult(BaseModel):
    username:        str
    prediction:      str
    confidence:      float
    top_2:           List[tuple]
    needs_attention: bool

# ── Helpers ───────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def run_inference(text: str) -> dict:
    """Run model inference and return prediction dict."""
    enc = tokenizer(
        clean_text(text),
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(
            input_ids=enc["input_ids"].to(DEVICE),
            attention_mask=enc["attention_mask"].to(DEVICE)
        ).logits

    probs   = torch.softmax(logits, dim=1)[0].cpu().numpy()
    top_ids = probs.argsort()[::-1][:2]

    return {
        "prediction" : ID2LABEL[top_ids[0]],
        "confidence" : round(float(probs[top_ids[0]]), 4),
        "top_2"      : [(ID2LABEL[i], round(float(probs[i]), 4))
                        for i in top_ids],
    }

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":  "ok",
        "model":   MODEL_DIR,
        "device":  str(DEVICE),
        "classes": list(ID2LABEL.values()),
        "alert_classes": list(ALERT_CLASSES)
    }

@app.post("/classify", response_model=List[str])
def classify(messages: List[UserMessage]):
    """
    Returns usernames whose messages are classified as
    any distress category (not Normal).
    """
    if not messages:
        raise HTTPException(status_code=400, detail="Empty message list.")

    at_risk = []
    for msg in messages:
        result = run_inference(msg.text)
        if result["prediction"] in ALERT_CLASSES:
            at_risk.append(msg.username)
    return at_risk

@app.post("/classify/detailed", response_model=List[DetailedResult])
def classify_detailed(messages: List[UserMessage]):
    """
    Returns full prediction details for every user in the batch.
    """
    if not messages:
        raise HTTPException(status_code=400, detail="Empty message list.")

    results = []
    for msg in messages:
        r = run_inference(msg.text)
        results.append(DetailedResult(
            username=msg.username,
            prediction=r["prediction"],
            confidence=r["confidence"],
            top_2=r["top_2"],
            needs_attention=r["prediction"] in ALERT_CLASSES
        ))
    return results

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("mental_app:app", host="0.0.0.0", port=8001, reload=False)
