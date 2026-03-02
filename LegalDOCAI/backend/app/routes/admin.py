from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from backend.app.core.deps import get_current_user
from backend.app.services.ml_service import classifier
import os, json, csv, time
from typing import List, Dict, Any

router = APIRouter(tags=["admin"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "backend", "data")
MODELS_DIR = os.path.join(BASE_DIR, "backend", "models")
DATASET_PATH = os.path.join(DATA_DIR, "training_data.json")
DATASET_META_PATH = os.path.join(DATA_DIR, "training_data.meta.json")
PIPELINE_PATH = os.path.join(MODELS_DIR, "document_model.pkl")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

PROJECT_ROOT = os.path.dirname(BASE_DIR)
AUTO_TRAIN_PATH = os.path.join(PROJECT_ROOT, "legal_dataset", "auto_training_data.json")
AUTO_TRAIN_STATS_PATH = os.path.join(PROJECT_ROOT, "legal_dataset", "auto_training_stats.json")


@router.get("/auto-training/stats")
async def auto_training_stats(user: dict = Depends(get_current_user)):
    stats: Dict[str, Any] = {
        "count": 0,
        "saved": 0,
        "last_added": None,
        "anomaly_skipped_count": 0,
        "fraud_skipped_count": 0,
        "low_conf_skipped_count": 0,
        "duplicate_skipped_count": 0,
    }
    data_count = 0
    if os.path.exists(AUTO_TRAIN_PATH):
        try:
            with open(AUTO_TRAIN_PATH, "r", encoding="utf-8") as f:
                payload = json.load(f) or []
            if isinstance(payload, list):
                data_count = len(payload)
        except Exception:
            data_count = 0
    if os.path.exists(AUTO_TRAIN_STATS_PATH):
        try:
            with open(AUTO_TRAIN_STATS_PATH, "r", encoding="utf-8") as f:
                loaded = json.load(f) or {}
            if isinstance(loaded, dict):
                stats.update(loaded)
        except Exception:
            pass
    stats["data_count"] = data_count
    return stats
