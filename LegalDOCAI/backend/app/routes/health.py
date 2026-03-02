import os
import time
from typing import Dict, Any

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from backend.app import database
from backend.app.core.config import OPENAI_MODEL
from backend.app.core.deps import is_openai_ready
from backend.app.services.ml_service import classifier

router = APIRouter()

_BOOT_TS = time.time()


def _db_status() -> Dict[str, Any]:
    if database.client is None:
        # Local JSON storage mode (DISABLE_MONGO=1 or Mongo unavailable)
        return {"mode": "local", "connected": True}
    try:
        database.client.admin.command("ping")
        return {"mode": "mongo", "connected": True}
    except Exception as exc:
        return {"mode": "mongo", "connected": False, "error": str(exc)}


@router.get("/health")
def health_check():
    """Liveness probe: process is up."""
    return {"status": "ok", "service": "legaldocai-backend"}


@router.get("/ready")
def readiness_check():
    """Readiness probe: dependencies needed for serving requests."""
    db = _db_status()
    checks = {
        "db": db,
        "ml_model_loaded": bool(classifier.is_loaded),
        "openai_ready": bool(is_openai_ready()),
    }
    ready = bool(db.get("connected")) and checks["ml_model_loaded"]
    return {
        "status": "ready" if ready else "degraded",
        "service": "legaldocai-backend",
        "checks": checks,
    }


@router.get("/metrics", response_class=PlainTextResponse)
def metrics():
    """Minimal Prometheus-style metrics for ops visibility."""
    db = _db_status()
    uptime = max(0.0, time.time() - _BOOT_TS)
    lines = [
        "# HELP legaldocai_uptime_seconds Process uptime in seconds",
        "# TYPE legaldocai_uptime_seconds gauge",
        f"legaldocai_uptime_seconds {uptime:.2f}",
        "# HELP legaldocai_db_connected Database connectivity (1=connected, 0=not connected)",
        "# TYPE legaldocai_db_connected gauge",
        f"legaldocai_db_connected {1 if db.get('connected') else 0}",
        "# HELP legaldocai_ml_model_loaded ML classifier availability (1=loaded, 0=not loaded)",
        "# TYPE legaldocai_ml_model_loaded gauge",
        f"legaldocai_ml_model_loaded {1 if classifier.is_loaded else 0}",
        "# HELP legaldocai_openai_ready OpenAI client readiness (1=ready, 0=not ready)",
        "# TYPE legaldocai_openai_ready gauge",
        f"legaldocai_openai_ready {1 if is_openai_ready() else 0}",
        "# HELP legaldocai_build_info Build and runtime labels",
        "# TYPE legaldocai_build_info gauge",
        f'legaldocai_build_info{{service="legaldocai-backend",model="{OPENAI_MODEL}",db_mode="{db.get("mode","unknown")}",pid="{os.getpid()}"}} 1',
    ]
    return "\n".join(lines) + "\n"
