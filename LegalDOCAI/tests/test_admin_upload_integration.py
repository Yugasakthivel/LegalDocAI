import io
import json
import types
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.testclient import TestClient


class InMemoryCollection:
    def __init__(self):
        self.items: List[Dict[str, Any]] = []

    def find_one(self, flt: Dict[str, Any], projection: Optional[Dict[str, int]] = None):
        for item in self.items:
            if all(item.get(k) == v for k, v in flt.items()):
                if not projection:
                    return dict(item)
                out = {}
                for key, include in projection.items():
                    if include:
                        out[key] = item.get(key)
                return out if out else dict(item)
        return None

    def insert_one(self, doc: Dict[str, Any]):
        self.items.append(dict(doc))

        class _Result:
            inserted_id = doc.get("job_id") or doc.get("doc_id") or doc.get("user_id")

        return _Result()

    def update_one(self, flt: Dict[str, Any], update: Dict[str, Any], upsert: bool = False):
        modified = 0
        for idx, item in enumerate(self.items):
            if all(item.get(k) == v for k, v in flt.items()):
                if "$set" in update:
                    merged = dict(item)
                    merged.update(update["$set"])
                    self.items[idx] = merged
                    modified = 1
                break
        if modified == 0 and upsert:
            new_doc = dict(flt)
            if "$set" in update:
                new_doc.update(update["$set"])
            self.items.append(new_doc)
            modified = 1

        class _Result:
            modified_count = modified

        return _Result()


def _install_fake_vectorstore():
    fake_vectorstore = types.ModuleType("backend.app.vectorstore")

    def _add_document(_doc):
        return None

    async def _add_document_async(_doc):
        return None

    def _fetch_document_by_id(_doc_id):
        return None

    def _search(*_args, **_kwargs):
        return []

    def _search_doc_first(*_args, **_kwargs):
        return []

    def _search_in_index(*_args, **_kwargs):
        return []

    def _get_chroma_collection(*_args, **_kwargs):
        return None

    def _embed_texts(*_args, **_kwargs):
        return []

    fake_vectorstore.add_document = _add_document
    fake_vectorstore.add_document_async = _add_document_async
    fake_vectorstore.fetch_document_by_id = _fetch_document_by_id
    fake_vectorstore.search = _search
    fake_vectorstore.search_doc_first = _search_doc_first
    fake_vectorstore.search_in_index = _search_in_index
    fake_vectorstore.get_chroma_collection = _get_chroma_collection
    fake_vectorstore.embed_texts = _embed_texts
    import sys

    sys.modules["backend.app.vectorstore"] = fake_vectorstore


def _build_app():
    _install_fake_vectorstore()
    from backend.app.routes import router as backend_api_router

    app = FastAPI()
    app.include_router(backend_api_router, prefix="/api")
    return app


def _setup_isolated_state(monkeypatch, tmp_path):
    _install_fake_vectorstore()

    import backend.app.core.deps as deps
    import backend.app.database as database
    import backend.app.routes.ai as ai_route
    import backend.app.routes.auth as auth_route
    import backend.app.routes.pipeline as pipeline_route
    import backend.app.routes.upload as upload_route
    import backend.app.services.storage_service as storage_service

    users = InMemoryCollection()
    docs = InMemoryCollection()
    jobs = InMemoryCollection()

    monkeypatch.setattr(auth_route, "users_collection", users)
    monkeypatch.setattr(deps, "users_collection", users)
    monkeypatch.setattr(database, "users_collection", users)

    monkeypatch.setattr(database, "documents_collection", docs)
    monkeypatch.setattr(storage_service, "documents_collection", docs)
    monkeypatch.setattr(upload_route, "collection", docs)
    monkeypatch.setattr(ai_route, "collection", docs)

    monkeypatch.setattr(pipeline_route, "pipeline_jobs_collection", jobs)
    monkeypatch.setattr(database, "pipeline_jobs_collection", jobs)

    monkeypatch.setattr(upload_route, "UPLOAD_FOLDER", str(tmp_path))
    monkeypatch.setattr(upload_route.ocr_service, "detect_file_type", lambda _name: "pdf")
    monkeypatch.setattr(upload_route.ocr_service, "resolve_ocr_lang_for_file", lambda *_args, **_kwargs: "eng")
    monkeypatch.setattr(upload_route.ocr_service, "extract_text_by_filetype", lambda *_args, **_kwargs: ["mock extracted text"])

    return users, docs, jobs


def _register_and_get_token(client: TestClient, email: str):
    payload = {
        "username": email.split("@")[0],
        "email": email,
        "password": "Test@12345",
    }
    res = client.post("/api/auth/register", json=payload)
    assert res.status_code == 200, res.text
    token = res.json()["access_token"]
    return token, payload


def test_admin_auto_training_stats_auth_and_shape(monkeypatch, tmp_path):
    _setup_isolated_state(monkeypatch, tmp_path)
    app = _build_app()
    client = TestClient(app)

    unauth = client.get("/api/auto-training/stats")
    assert unauth.status_code == 401

    token, _ = _register_and_get_token(client, "admin_stats@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    import backend.app.routes.admin as admin_route

    data_path = tmp_path / "auto_training_data.json"
    stats_path = tmp_path / "auto_training_stats.json"
    data_path.write_text(json.dumps([{"x": 1}, {"y": 2}]), encoding="utf-8")
    stats_path.write_text(
        json.dumps(
            {
                "count": 3,
                "saved": 1,
                "last_added": 123456,
                "anomaly_skipped_count": 2,
                "fraud_skipped_count": 1,
                "low_conf_skipped_count": 4,
                "duplicate_skipped_count": 5,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(admin_route, "AUTO_TRAIN_PATH", str(data_path))
    monkeypatch.setattr(admin_route, "AUTO_TRAIN_STATS_PATH", str(stats_path))

    res = client.get("/api/auto-training/stats", headers=headers)
    assert res.status_code == 200, res.text
    payload = res.json()

    required = {
        "count",
        "saved",
        "last_added",
        "anomaly_skipped_count",
        "fraud_skipped_count",
        "low_conf_skipped_count",
        "duplicate_skipped_count",
        "data_count",
    }
    assert required.issubset(payload.keys())
    assert payload["data_count"] == 2
    assert isinstance(payload["count"], int)
    assert isinstance(payload["saved"], int)
    assert payload["fraud_skipped_count"] == 1


def test_upload_includes_anomaly_and_fraud_fields(monkeypatch, tmp_path):
    _setup_isolated_state(monkeypatch, tmp_path)
    app = _build_app()
    client = TestClient(app)

    token, _ = _register_and_get_token(client, "upload_flags@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    import backend.app.routes.upload as upload_route

    async def fake_run_full_pipeline(doc_id: str, background_tasks=None):
        return {
            "doc_id": doc_id,
            "analytics": {
                "anomaly_detected": True,
                "anomaly_reason": "synthetic anomaly",
                "fraud_flag": False,
                "fraud_reason": "",
            },
            "results_summary": {"entities": {"names": []}},
        }

    monkeypatch.setattr(upload_route, "run_full_pipeline", fake_run_full_pipeline)

    res = client.post(
        "/api/upload",
        headers=headers,
        files={"file": ("sample.pdf", io.BytesIO(b"%PDF-1.4 mock"), "application/pdf")},
    )
    assert res.status_code == 200, res.text
    payload = res.json()
    analytics = payload.get("analytics") or {}

    for key in ("anomaly_detected", "anomaly_reason", "fraud_flag", "fraud_reason"):
        assert key in analytics
    assert analytics["anomaly_detected"] is True
    assert analytics["fraud_flag"] is False


def test_non_legal_pipeline_risk_clauses_count_zero(monkeypatch, tmp_path):
    _setup_isolated_state(monkeypatch, tmp_path)
    app = _build_app()
    client = TestClient(app)

    token, _ = _register_and_get_token(client, "non_legal@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    import backend.app.services.storage_service as storage_service
    import backend.app.services.analysis_service as analysis_service
    import backend.app.services.ml_service as ml_service

    doc_id = "doc-non-legal"
    storage_service.create_initial_document(
        doc_id=doc_id,
        filename="resume.txt",
        file_type="txt",
        text="John Doe\nSkills: Python, SQL\nExperience: 3 years\nEducation: BSc",
    )

    monkeypatch.setattr(ml_service.classifier, "detect_document_type_by_rules", lambda _t: "Resume")
    monkeypatch.setattr(ml_service.classifier, "predict_document_type", lambda _t: ("Resume", 0.95))
    monkeypatch.setattr(
        analysis_service,
        "classify_legal_two_step",
        lambda _t, _doc_type=None: {"is_legal": False, "is_indian": False, "category": "Non-Legal Document"},
    )

    res = client.post("/api/pipeline/full", headers=headers, data={"doc_id": doc_id})
    assert res.status_code == 200, res.text
    payload = res.json()
    analytics = payload.get("analytics") or {}
    assert analytics.get("risk_clauses_count") == 0
