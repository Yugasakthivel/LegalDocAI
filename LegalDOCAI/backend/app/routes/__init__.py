from fastapi import APIRouter
from .health import router as health_router
from .upload import router as upload_router
from .search import router as search_router
from .document import router as document_router

# API Router with common prefix
router = APIRouter(prefix="/api")

# Attach sub-routers
router.include_router(health_router, prefix="", tags=["health"])
router.include_router(upload_router, prefix="", tags=["upload"])
router.include_router(search_router, prefix="", tags=["search"])
router.include_router(search_router, prefix="", tags=["process"])
router.include_router(document_router, prefix="", tags=["document"])
