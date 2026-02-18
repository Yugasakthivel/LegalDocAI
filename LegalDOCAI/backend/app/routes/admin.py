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

 
