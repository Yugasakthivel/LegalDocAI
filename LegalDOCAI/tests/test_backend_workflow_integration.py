import asyncio
import io
import sys
import time
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


def _build_app():
    _install_fake_vectorstore()

    from backend.app.routes import router as backend_api_router

    app = FastAPI()
    app.include_router(backend_api_router, prefix="/api")
    return app


def _setup_isolated_state(monkeypatch, tmp_path):
    _install_fake_vectorstore()

    import backend.app.core.deps as deps
    import backend.app.routes.ai as ai_route
    import backend.app.routes.auth as auth_route
    import backend.app.routes.pipeline as pipeline_route
    import backend.app.routes.upload as upload_route

    users = InMemoryCollection()
    docs = InMemoryCollection()
    jobs = InMemoryCollection()

    monkeypatch.setattr(auth_route, "users_collection", users)
    monkeypatch.setattr(deps, "users_collection", users)
    monkeypatch.setattr(upload_route, "collection", docs)
    monkeypatch.setattr(ai_route, "collection", docs)
    monkeypatch.setattr(pipeline_route, "pipeline_jobs_collection", jobs)
    monkeypatch.setattr(upload_route, "UPLOAD_FOLDER", str(tmp_path))

    monkeypatch.setattr(upload_route.ocr_service, "detect_file_type", lambda _name: "pdf")
    monkeypatch.setattr(upload_route.ocr_service, "resolve_ocr_lang_for_file", lambda *_args, **_kwargs: "eng")
    monkeypatch.setattr(upload_route.ocr_service, "extract_text_by_filetype", lambda *_args, **_kwargs: ["mock extracted text"])

    async def fake_run_full_pipeline(doc_id: str, background_tasks=None):
        return {
            "doc_id": doc_id,
            "results_summary": {"entities": {"names": []}},
            "analytics": {"document_type": "Mock", "legality_score": 80, "ai_confidence": 90},
        }

    monkeypatch.setattr(upload_route, "run_full_pipeline", fake_run_full_pipeline)

    return users, docs, jobs


def _install_fake_vectorstore():
    # Avoid importing heavy vector db dependencies during route import.
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
    sys.modules["backend.app.vectorstore"] = fake_vectorstore


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


def test_auth_upload_start_status_success(monkeypatch, tmp_path):
    _setup_isolated_state(monkeypatch, tmp_path)
    app = _build_app()
    client = TestClient(app)

    token, user = _register_and_get_token(client, "workflow_user@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    upload = client.post(
        "/api/upload",
        headers=headers,
        files={"file": ("sample.pdf", io.BytesIO(b"%PDF-1.4 mock"), "application/pdf")},
    )
    assert upload.status_code == 200, upload.text
    doc_id = upload.json().get("doc_id")
    assert doc_id

    start_analysis = client.post(
        "/api/start-analysis",
        json={"email": user["email"], "uploadedDocumentId": doc_id},
    )
    assert start_analysis.status_code == 200, start_analysis.text
    assert start_analysis.json().get("docId") == doc_id

    import backend.app.routes.pipeline as pipeline_route

    async def fake_pipeline_success(job_id: str, start_doc_id: str):
        pipeline_route._update_pipeline_job(
            job_id,
            state="completed",
            result={"doc_id": start_doc_id, "analytics": {"legality_score": 77}},
            error=None,
        )

    monkeypatch.setattr(pipeline_route, "_run_pipeline_job", fake_pipeline_success)

    start = client.post("/api/pipeline/start", headers=headers, data={"doc_id": doc_id})
    assert start.status_code == 200, start.text
    job_id = start.json()["job_id"]

    final = None
    for _ in range(40):
        status = client.get(f"/api/pipeline/status/{job_id}", headers=headers)
        assert status.status_code == 200, status.text
        payload = status.json()
        if payload.get("state") == "completed":
            final = payload
            break
        time.sleep(0.02)

    assert final is not None
    assert final["state"] == "completed"
    assert final["result"]["doc_id"] == doc_id


def test_pipeline_status_failure_case(monkeypatch, tmp_path):
    _setup_isolated_state(monkeypatch, tmp_path)
    app = _build_app()
    client = TestClient(app)

    token, _ = _register_and_get_token(client, "pipeline_failure@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    import backend.app.routes.pipeline as pipeline_route

    async def fake_pipeline_failure(job_id: str, _doc_id: str):
        pipeline_route._update_pipeline_job(
            job_id,
            state="failed",
            result=None,
            error="simulated pipeline error",
        )

    monkeypatch.setattr(pipeline_route, "_run_pipeline_job", fake_pipeline_failure)

    start = client.post("/api/pipeline/start", headers=headers, data={"doc_id": "doc-failure-case"})
    assert start.status_code == 200, start.text
    job_id = start.json()["job_id"]

    final = None
    for _ in range(40):
        status = client.get(f"/api/pipeline/status/{job_id}", headers=headers)
        assert status.status_code == 200, status.text
        payload = status.json()
        if payload.get("state") == "failed":
            final = payload
            break
        time.sleep(0.02)

    assert final is not None
    assert final["state"] == "failed"
    assert "simulated pipeline error" in (final.get("error") or "")


def test_auth_login_failure_wrong_password(monkeypatch, tmp_path):
    _setup_isolated_state(monkeypatch, tmp_path)
    app = _build_app()
    client = TestClient(app)

    _register_and_get_token(client, "wrong_pass_user@example.com")
    bad_login = client.post(
        "/api/auth/login",
        json={"email": "wrong_pass_user@example.com", "password": "WrongPass@123"},
    )
    assert bad_login.status_code == 401


def test_unauthorized_access_for_upload_and_pipeline(monkeypatch, tmp_path):
    _setup_isolated_state(monkeypatch, tmp_path)
    app = _build_app()
    client = TestClient(app)

    upload_no_auth = client.post(
        "/api/upload",
        files={"file": ("sample.pdf", io.BytesIO(b"%PDF-1.4 mock"), "application/pdf")},
    )
    assert upload_no_auth.status_code == 401

    pipeline_start_no_auth = client.post("/api/pipeline/start", data={"doc_id": "unauth-doc"})
    assert pipeline_start_no_auth.status_code == 401

    pipeline_status_no_auth = client.get("/api/pipeline/status/some-job-id")
    assert pipeline_status_no_auth.status_code == 401


def test_chat_rag_missing_doc_falls_back_to_ai(monkeypatch, tmp_path):
    _setup_isolated_state(monkeypatch, tmp_path)
    app = _build_app()
    client = TestClient(app)

    token, _ = _register_and_get_token(client, "chat_fallback_user@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    # Missing doc_id should not hard-fail; it should use fallback mode.
    res = client.post("/api/chat-rag", headers=headers, params={"doc_id": "missing-doc", "question": "What is the owner name?"})
    assert res.status_code == 200, res.text
    payload = res.json()
    assert payload.get("mode") == "ai_fallback"
    assert payload.get("fallback_used") is True
    assert "answer" in payload


def test_chat_rag_uses_global_rag_when_doc_missing(monkeypatch, tmp_path):
    _setup_isolated_state(monkeypatch, tmp_path)
    app = _build_app()
    client = TestClient(app)

    token, _ = _register_and_get_token(client, "chat_global_rag_user@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    import backend.app.routes.ai as ai_route

    monkeypatch.setattr(
        ai_route.vectorstore,
        "search",
        lambda _q, top_k=5: [{"doc_id": "global-doc-1", "score": 0.92}],
    )
    monkeypatch.setattr(
        ai_route.vectorstore,
        "fetch_document_by_id",
        lambda _doc_id: {"doc_id": "global-doc-1", "combined_text": "Owner name is Raj Kumar. Survey number is 12."},
    )

    res = client.post(
        "/api/chat-rag",
        headers=headers,
        params={"doc_id": "missing-doc", "question": "Who is the owner name?"},
    )
    assert res.status_code == 200, res.text
    payload = res.json()
    assert payload.get("mode") == "rag_global"
    assert payload.get("fallback_used") is True
    assert payload.get("doc_found") is False
    assert "answer" in payload
