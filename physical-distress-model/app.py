"""
Emergency Detection Microservice
=================================
POST /classify
  Input  : [{"username": "user1", "text": "I feel like giving up."}, ...]
  Output : ["user1", "user3"]   -- only at-risk usernames

POST /classify/detailed
  Input  : same as above
  Output : [{"username": "user1", "probability": 0.87, "emergency": true}, ...]

GET  /health
  Output : {"status": "ok", "model": "bert-base-uncased", "device": "cuda/cpu"}
"""

import re
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from transformers import BertTokenizer, BertForSequenceClassification
import uvicorn

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_DIR  = "bert_emergency_model"   # folder saved by Colab (HuggingFace format)
MAX_LEN    = 128
THRESHOLD  = 0.50                     # raise to 0.6-0.7 to reduce false alarms
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load model once at startup ────────────────────────────────────────────────
print(f"Loading model from '{MODEL_DIR}' on {DEVICE}...")
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model     = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(DEVICE).eval()
print("✅ Model ready")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Emergency Detection Service",
    description="Classifies sentences as emergency or non-emergency using BERT.",
    version="1.0.0"
)

# Allow Spring Boot backend to call this service
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # restrict to your Spring Boot host in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────────────────────
class UserMessage(BaseModel):
    username: str
    text: str

class DetailedResult(BaseModel):
    username:    str
    probability: float
    emergency:   bool

# ── Helpers ───────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)       # remove URLs
    text = re.sub(r"<.*?>", "", text)                 # remove HTML
    text = re.sub(r"@\w+", "", text)                  # remove mentions
    text = re.sub(r"#(\w+)", r"\1", text)             # keep hashtag word
    text = re.sub(r"[^\x00-\x7F]+", " ", text)        # remove non-ASCII
    return re.sub(r"\s+", " ", text).strip()

def get_probability(text: str) -> float:
    """Run BERT inference and return P(emergency)."""
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
            attention_mask=enc["attention_mask"].to(DEVICE),
            token_type_ids=enc["token_type_ids"].to(DEVICE)
        ).logits
    return torch.softmax(logits, dim=1)[0][1].item()

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model":  MODEL_DIR,
        "device": str(DEVICE),
        "threshold": THRESHOLD
    }

@app.post("/classify", response_model=List[str])
def classify(messages: List[UserMessage]):
    """
    Returns only the usernames whose messages are classified as emergency.
    This is the primary endpoint for the Spring Boot backend.
    """
    if not messages:
        raise HTTPException(status_code=400, detail="Message list cannot be empty.")
    return [
        msg.username
        for msg in messages
        if get_probability(msg.text) >= THRESHOLD
    ]

@app.post("/classify/detailed", response_model=List[DetailedResult])
def classify_detailed(messages: List[UserMessage]):
    """
    Returns probability scores for ALL users — useful for dashboards or logging.
    """
    if not messages:
        raise HTTPException(status_code=400, detail="Message list cannot be empty.")
    results = []
    for msg in messages:
        prob = get_probability(msg.text)
        results.append(DetailedResult(
            username=msg.username,
            probability=round(prob, 4),
            emergency=prob >= THRESHOLD
        ))
    return results

# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
