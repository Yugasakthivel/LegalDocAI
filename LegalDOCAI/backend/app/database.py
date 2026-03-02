import os
import json
import threading
from pymongo import MongoClient, ASCENDING
from backend.app.core.config import MONGO_URI, DB_NAME

class LocalCollection:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                json.dump([], f)

    def _get_by_path(self, doc: dict, path: str):
        try:
            cur = doc
            for part in (path or "").split("."):
                if not part:
                    return None
                if not isinstance(cur, dict):
                    return None
                cur = cur.get(part)
            return cur
        except Exception:
            return None

    def _set_by_path(self, doc: dict, path: str, value):
        try:
            parts = (path or "").split(".")
            cur = doc
            for part in parts[:-1]:
                if not part:
                    return
                nxt = cur.get(part)
                if not isinstance(nxt, dict):
                    nxt = {}
                    cur[part] = nxt
                cur = nxt
            last = parts[-1] if parts else ""
            if last:
                cur[last] = value
        except Exception:
            return

    def _load(self):
        with self._lock:
            with open(self.path, "r") as f:
                return json.load(f)
    def _save(self, data):
        with self._lock:
            with open(self.path, "w") as f:
                json.dump(data, f)
    def find_one(self, filter, projection=None):
        data = self._load()
        for d in data:
            ok = True
            for k, v in filter.items():
                if d.get(k) != v:
                    ok = False
                    break
            if ok:
                if projection:
                    out = {}
                    for pk, pv in projection.items():
                        if pv:
                            out[pk] = self._get_by_path(d, pk) if "." in pk else d.get(pk)
                    return out if out else d
                return d
        return None
    def insert_one(self, doc):
        data = self._load()
        data.append(doc)
        self._save(data)
        class R:
            inserted_id = doc.get("user_id")
        return R()

    def find(self, filter=None, projection=None):
        data = self._load()
        filter = filter or {}
        out = []
        for d in data:
            ok = True
            for k, v in filter.items():
                if d.get(k) != v:
                    ok = False
                    break
            if not ok:
                continue

            if projection:
                proj_doc = {}
                for pk, pv in projection.items():
                    if pv:
                        proj_doc[pk] = self._get_by_path(d, pk) if "." in pk else d.get(pk)
                out.append(proj_doc)
            else:
                out.append(d)
        return out

    def update_one(self, filter, update, upsert=False):
        data = self._load()
        modified = 0
        for i, d in enumerate(data):
            ok = True
            for k, v in filter.items():
                if d.get(k) != v:
                    ok = False
                    break
            if ok:
                if "$set" in update:
                    for k2, v2 in update["$set"].items():
                        if "." in k2:
                            self._set_by_path(d, k2, v2)
                        else:
                            d[k2] = v2
                    data[i] = d
                    modified = 1
                    break
        if modified == 0 and upsert:
            new_doc = dict(filter)
            if "$set" in update:
                for k2, v2 in update["$set"].items():
                    if "." in k2:
                        self._set_by_path(new_doc, k2, v2)
                    else:
                        new_doc[k2] = v2
            data.append(new_doc)
            modified = 1
        self._save(data)
        class R:
            modified_count = modified
        return R()
    def delete_one(self, filter):
        data = self._load()
        deleted = 0
        for i, d in enumerate(data):
            ok = True
            for k, v in filter.items():
                if d.get(k) != v:
                    ok = False
                    break
            if ok:
                del data[i]
                deleted = 1
                break
        if deleted:
            self._save(data)
        class R:
            deleted_count = deleted
        return R()
    def create_index(self, *args, **kwargs):
        return None

_disable_mongo = os.getenv("DISABLE_MONGO", "1").strip() == "1"
if _disable_mongo:
    client = None
    db = None
    documents_collection = LocalCollection(os.path.join(os.getcwd(), "local_db_documents.json"))
    users_collection = LocalCollection(os.path.join(os.getcwd(), "local_db_users.json"))
    pipeline_jobs_collection = LocalCollection(os.path.join(os.getcwd(), "local_db_pipeline_jobs.json"))
else:
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        client.admin.command("ping")
        db = client[DB_NAME]
        documents_collection = db["documents"]
        users_collection = db["users"]
        pipeline_jobs_collection = db["pipeline_jobs"]
        try:
            documents_collection.create_index([("doc_id", ASCENDING)], unique=True)
        except Exception:
            pass
        try:
            pipeline_jobs_collection.create_index([("job_id", ASCENDING)], unique=True)
        except Exception:
            pass
    except Exception:
        client = None
        db = None
        documents_collection = LocalCollection(os.path.join(os.getcwd(), "local_db_documents.json"))
        users_collection = LocalCollection(os.path.join(os.getcwd(), "local_db_users.json"))
        pipeline_jobs_collection = LocalCollection(os.path.join(os.getcwd(), "local_db_pipeline_jobs.json"))
