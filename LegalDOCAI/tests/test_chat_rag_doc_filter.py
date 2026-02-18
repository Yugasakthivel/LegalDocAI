import json
import uuid
from fastapi.testclient import TestClient
from main import app
from backend.app.core.deps import get_current_user
from backend.app.database import documents_collection as collection

app.dependency_overrides[get_current_user] = lambda: {"email": "tests@example.com"}
client = TestClient(app)

def _register():
    em = f"t_{uuid.uuid4().hex[:8]}@example.com"
    r = client.post("/api/auth/register", json={"username":"t","email":em,"password":"Passw0rd!"})
    if r.status_code == 200 and r.json().get("access_token"):
        return r.json()["access_token"]
    lr = client.post("/api/auth/login", json={"email":em,"password":"Passw0rd!"})
    return lr.json()["access_token"]

def test_chat_rag_returns_only_active_doc_sources():
    d1 = {"doc_id":"docA","filename":"a.txt","combined_text":"This is A document content about sale deed."}
    d2 = {"doc_id":"docB","filename":"b.txt","combined_text":"This is B document content about lease agreement."}
    collection.update_one({"doc_id": d1["doc_id"]}, {"$set": d1}, upsert=True)
    collection.update_one({"doc_id": d2["doc_id"]}, {"$set": d2}, upsert=True)
    r1 = client.post("/api/chat-rag?doc_id=docA&question=What%20is%20this%20document%3F")
    assert r1.status_code == 200
    sources = r1.json().get("sources") or []
    for s in sources:
        assert s.get("doc_id") == "docA"
