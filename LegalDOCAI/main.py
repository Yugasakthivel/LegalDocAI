import os
import shutil
from contextlib import asynccontextmanager
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from backend.app.routes import router as backend_api_router
from backend.app.core.config import (
    OPENAI_MODEL, MONGO_URI, DB_NAME, TESSERACT_CMD, UPLOAD_FOLDER
)
from backend.app.core.deps import get_openai_client
from backend.app.database import documents_collection as collection
from backend.app.services import ocr_service, analysis_service
from backend.app.services.risk_service import compute_combined_risk_index
from backend.app.services.verification_service import verify_document, validate_signers
from backend.app.services.compliance_service import jurisdiction_checks, curated_checks
from bson import ObjectId

from backend.app.services.ml_service import classifier
import json

# Lifespan context manager (replaces deprecated on_event)
@asynccontextmanager
async def lifespan(app):
    # Startup
    print("===============================================")
    print("LegalDocAI Backend Started")
    print(f"Model: {OPENAI_MODEL}")
    print(f"Mongo: {MONGO_URI}")
    print(f"DB: {DB_NAME}")
    print(f"Tesseract: {TESSERACT_CMD}")
    
    # Auto-Train ML Model if missing
    if not classifier.is_loaded:
        print("ML Model not found. Attempting to auto-train...")
        try:
            # Path to training data relative to main.py
            # LegalDOCAI/main.py -> ../legal_dataset/training_data.json
            base_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.abspath(os.path.join(base_dir, "..", "legal_dataset", "training_data.json"))
            
            if os.path.exists(data_path):
                with open(data_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                texts = [d["text"] for d in data]
                labels = [d["category"] for d in data]
                
                print(f"Found {len(texts)} training samples. Training now...")
                result = classifier.train_model(texts, labels)
                if result.get("success"):
                    print("Auto-training successful: Model saved.")
                else:
                    print(f"Auto-training failed: {result.get('error')}")
            else:
                print(f"Training data not found at {data_path}")
        except Exception as e:
            print(f"Error during auto-training: {e}")
    else:
        print("ML Model is loaded and ready.")

    print("===============================================")
    yield
    # Shutdown
    print("===============================================")
    print("🛑 LegalDocAI Backend Shutting Down")
    print("===============================================")

app = FastAPI(title="⚖️ LegalDocAI Backend", version="1.0.0", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(backend_api_router, prefix="/api")

# Static UI
try:
    os.makedirs("web", exist_ok=True)
    app.mount("/ui", StaticFiles(directory="web", html=True), name="ui")
except Exception:
    pass

# Mount skeleton app (legaldocai_skeleton) under /skeleton
try:
    import sys
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if base_dir not in sys.path:
        sys.path.append(base_dir)
        
    # Add legaldocai_skeleton to sys.path so 'import app' works within it
    skeleton_dir = os.path.join(base_dir, "legaldocai_skeleton")
    if skeleton_dir not in sys.path:
        sys.path.append(skeleton_dir)

    from legaldocai_skeleton.app.main import app as skeleton_app
    app.mount("/skeleton", skeleton_app)
    print("Mounted legaldocai_skeleton at /skeleton")
except Exception as e:
    print(f"Failed to mount legaldocai_skeleton: {e}")

# Removed legacy routes


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
