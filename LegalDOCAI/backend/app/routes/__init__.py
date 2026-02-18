from . import health, upload, analysis, dataset, report, ai, document, documents, auth, search, modules, pipeline, admin, trust_safety
from fastapi import APIRouter

router = APIRouter()

router.include_router(health.router)
router.include_router(auth.router)
router.include_router(upload.router)
router.include_router(analysis.router, prefix="/analysis")
router.include_router(dataset.router, prefix="/dataset")
router.include_router(report.router, prefix="/report")
router.include_router(ai.router)
router.include_router(document.router)
router.include_router(documents.router)
router.include_router(search.router, prefix="/search")
router.include_router(modules.router)
router.include_router(pipeline.router)
router.include_router(admin.router, prefix="/admin")
router.include_router(trust_safety.router)
