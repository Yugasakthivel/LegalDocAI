import sys
import os
import json
import hashlib
from collections import Counter
from typing import Dict, List

# Add project root to sys.path so we can import app modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from backend.app.services.ml_service import classifier

DATA_FILE = os.path.join(project_root, "legal_dataset", "training_data.json")
AUTO_DATA_FILE = os.path.join(project_root, "legal_dataset", "auto_training_data.json")
MERGED_DATA_FILE = os.path.join(project_root, "legal_dataset", "merged_training_data.json")


def _safe_load_list(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f) or []
        return payload if isinstance(payload, list) else []
    except Exception:
        return []


def _normalize_record(item: Dict) -> Dict:
    if not isinstance(item, dict):
        return {"text": "", "category": "Unknown"}
    txt = (item.get("text") or "").strip()
    cat_raw = (item.get("category") or item.get("label") or "Unknown").strip() or "Unknown"
    cat = classifier.canonicalize_category(cat_raw)
    return {"text": txt, "category": cat}


def _merge_dedup(base_data: List[Dict], auto_data: List[Dict]) -> List[Dict]:
    merged: List[Dict] = []
    seen = set()
    for raw in (base_data or []) + (auto_data or []):
        rec = _normalize_record(raw)
        if not rec["text"]:
            continue
        key = hashlib.sha1(f"{rec['category']}::{rec['text']}".encode("utf-8", errors="ignore")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        merged.append(rec)
    return merged


def main():
    print("Starting Model Training Script")

    if not os.path.exists(DATA_FILE):
        print(f"Data file not found at {DATA_FILE}")
        print("Please run generate_synthetic_data.py first.")
        return

    try:
        base_data = _safe_load_list(DATA_FILE)
        auto_data = _safe_load_list(AUTO_DATA_FILE)
        data = _merge_dedup(base_data, auto_data)

        os.makedirs(os.path.dirname(MERGED_DATA_FILE), exist_ok=True)
        with open(MERGED_DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        texts = [d.get("text", "") for d in data if d.get("text")]
        labels = [d.get("category", "Unknown") for d in data if d.get("text")]

        print(
            f"Loaded {len(texts)} samples "
            f"(base={len(base_data)}, auto={len(auto_data)}, merged_dedup={len(data)})."
        )
        print(f"Merged dataset saved: {MERGED_DATA_FILE}")

        dist = Counter(labels)
        print("Class distribution:")
        for cls, cnt in sorted(dist.items(), key=lambda x: (-x[1], x[0])):
            print(f" - {cls}: {cnt}")
        print("Total dataset size:", len(texts))

        result = classifier.train_model(texts, labels)
        print(result)

        # Quick load check and sample prediction + anomaly verification
        print("Verifying model load and sample prediction...")
        print("is_loaded:", classifier.is_loaded)
        sample = "This SALE DEED is made between the Vendor and the Vendee regarding the schedule property."
        res = classifier.predict_document_type(sample)
        pred, conf = res[0], res[1]
        print(f"Sample prediction: {pred} (confidence: {int(conf * 100)}%)")
        an, reason = classifier.detect_anomaly(sample, pred, int(conf * 100))
        print(f"Sample anomaly_detected: {an}, anomaly_reason: {reason}")

    except Exception as e:
        print(f"Error during training: {e}")


if __name__ == "__main__":
    main()
