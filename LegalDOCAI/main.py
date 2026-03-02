import os
import shutil
import time
import uuid
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Form, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from backend.app.routes import router as backend_api_router
from backend.app.routes.verify import router as verify_router
from backend.app.core.config import (
    OPENAI_MODEL, MONGO_URI, DB_NAME, TESSERACT_CMD, UPLOAD_FOLDER
)
from backend.app.core.deps import get_openai_client, init_openai_from_env
from backend.app.services.risk_service import compute_combined_risk_index
from backend.app.services.verification_service import verify_document, validate_signers
from backend.app.services.compliance_service import jurisdiction_checks, curated_checks
from bson import ObjectId

from backend.app.services.ml_service import classifier
import json

logger = logging.getLogger("legaldocai")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )

# Lifespan context manager (replaces deprecated on_event)
@asynccontextmanager
async def lifespan(app):
    # Startup
    init_openai_from_env()
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
            base_dir = os.path.dirname(os.path.abspath(__file__))
            # Prefer last uploaded dataset if present
            last_dataset_file = os.path.abspath(os.path.join(base_dir, "data", "datasets", "last_dataset_path.txt"))
            dataset_path = None
            if os.path.exists(last_dataset_file):
                try:
                    with open(last_dataset_file, "r", encoding="utf-8") as f:
                        candidate = (f.read() or "").strip()
                    if candidate and os.path.exists(candidate):
                        dataset_path = candidate
                except Exception:
                    dataset_path = None

            # Fallback to repo default dataset
            if not dataset_path:
                dataset_path = os.path.abspath(os.path.join(base_dir, "..", "legal_dataset", "training_data.json"))

            if os.path.exists(dataset_path):
                with open(dataset_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Accept either {text,label} or {text,category}
                texts = [d.get("text") for d in data if isinstance(d, dict)]
                labels = [d.get("label") or d.get("category") for d in data if isinstance(d, dict)]
                pairs = [(t, l) for t, l in zip(texts, labels) if t and l]
                texts = [t for t, _ in pairs]
                labels = [l for _, l in pairs]

                print(f"Found {len(texts)} training samples. Training now...")
                result = classifier.train_model(texts, labels)
                if result.get("success"):
                    print("Auto-training successful: Model saved.")
                else:
                    print(f"Auto-training failed: {result.get('error')}")
            else:
                print(f"Training data not found at {dataset_path}")
        except Exception as e:
            print(f"Error during auto-training: {e}")
    else:
        print("ML Model is loaded and ready.")

    print("===============================================")
    yield
    # Shutdown
    print("===============================================")
    print("ðŸ›‘ LegalDocAI Backend Shutting Down")
    print("===============================================")

app = FastAPI(title="âš–ï¸ LegalDocAI Backend", version="1.0.0", lifespan=lifespan)

@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - started) * 1000.0
        logger.exception(
            "request_failed request_id=%s method=%s path=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            duration_ms,
        )
        raise

    duration_ms = (time.perf_counter() - started) * 1000.0
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_completed request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response

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
app.include_router(verify_router, prefix="/api/verify")

# Static UI
try:
    os.makedirs("web", exist_ok=True)
    app.mount("/ui", StaticFiles(directory="web", html=True), name="ui")
except Exception:
    pass

try:
    enable_skeleton = os.getenv("MOUNT_SKELETON", "false").lower() == "true"
    if enable_skeleton:
        import sys
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if base_dir not in sys.path:
            sys.path.append(base_dir)
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
    import socket
    import uvicorn

    def _find_available_port(host: str, preferred_port: int, max_tries: int = 50) -> int:
        for candidate in range(preferred_port, preferred_port + max_tries):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind((host, candidate))
                return candidate
            except OSError:
                continue
            finally:
                sock.close()
        raise RuntimeError(f"No free port found in range {preferred_port}-{preferred_port + max_tries - 1}")

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload_flag = os.getenv("UVICORN_RELOAD", "false").lower() == "true"

    try:
        uvicorn.run(app, host=host, port=port, reload=reload_flag)
    except OSError as e:
        if getattr(e, "errno", None) in (10048, 98):
            fallback_port = _find_available_port(host, port + 1)
            print(f"Port {port} is in use. Starting on fallback port {fallback_port}.")
            uvicorn.run(app, host=host, port=fallback_port, reload=reload_flag)
        else:
            raise
