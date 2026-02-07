import os
import logging
from pathlib import Path
from typing import List

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("score_headlines_api")


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Headline Sentiment Scoring Service")


# -----------------------------
# Model loading (IMPORTANT: load once)
# -----------------------------
def load_embedder() -> SentenceTransformer:
    """
    Prefer local path on the linux server to avoid large downloads.
    Fallback to model name if needed.
    """
    local_model_path = "/opt/huggingface_models/all-MiniLM-L6-v2"
    try:
        logger.info("Loading embedder from local path: %s", local_model_path)
        return SentenceTransformer(local_model_path)
    except Exception:
        logger.warning("Local embedder load failed; falling back to model name.")
        return SentenceTransformer("all-MiniLM-L6-v2")


def load_svm_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


# Global singletons (loaded once)
try:
    EMBEDDER = load_embedder()
    SVM_MODEL = load_svm_model(Path("svm.joblib"))
    logger.info("Models loaded successfully (embedder + svm).")
except Exception as e:
    logger.critical("Failed to load models at startup: %s", str(e), exc_info=True)
    raise


# -----------------------------
# Request/Response Schemas
# -----------------------------
class ScoreHeadlinesRequest(BaseModel):
    headlines: List[str] = Field(..., description="List of headline strings to score.")


class ScoreHeadlinesResponse(BaseModel):
    labels: List[str] = Field(..., description="Predicted labels for each headline.")


# -----------------------------
# Routes
# -----------------------------
@app.get("/status")
def status():
    return {"status": "OK"}


@app.post("/score_headlines", response_model=ScoreHeadlinesResponse)
def score_headlines(payload: ScoreHeadlinesRequest):
    n = len(payload.headlines) if payload.headlines is not None else 0
    logger.info("Received /score_headlines request. n_headlines=%d", n)

    if payload.headlines is None or n == 0:
        logger.warning("Empty headlines list.")
        raise HTTPException(status_code=400, detail="headlines must be a non-empty list of strings")

    cleaned = []
    for i, h in enumerate(payload.headlines):
        if not isinstance(h, str):
            logger.warning("Non-string headline at index %d.", i)
            raise HTTPException(status_code=400, detail=f"headline at index {i} is not a string")
        s = h.strip()
        if not s:
            logger.warning("Empty/blank headline at index %d.", i)
            raise HTTPException(status_code=400, detail=f"headline at index {i} is empty")
        cleaned.append(s)

    try:
        embeddings = EMBEDDER.encode(cleaned)
        preds = SVM_MODEL.predict(embeddings)
        labels = [str(x) for x in preds]
    except Exception as e:
        logger.error("Scoring failed: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error while scoring headlines")

    if len(labels) != len(cleaned):
        logger.error("Prediction size mismatch. labels=%d headlines=%d", len(labels), len(cleaned))
        raise HTTPException(status_code=500, detail="Prediction size mismatch")

    # IMPORTANT: do not return original headline text
    return {"labels": labels}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)
