import os
import joblib
import numpy as np
import re
import json
import csv
import warnings
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from typing import Dict, Any, List, Tuple, Iterable, Optional, Callable

# Constants
# Path: backend/app/services/ml_service.py
# Goal: Reach LegalDOCAI/data/models
# From services: .. (app) -> .. (backend) -> .. (LegalDOCAI) -> data/models
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODEL_DIR = os.path.join(BASE_DIR, "data", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "doc_classifier.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
ALT_MODELS_DIR = os.path.join(BASE_DIR, "backend", "models")
ALT_PIPELINE_PATH = os.path.join(ALT_MODELS_DIR, "document_model.pkl")

DOC_TYPE_RULES = [
    {"label": "Patta", "all": [], "any": ["patta", "land patta", "patta passbook", "ryotwari patta"]},
    {"label": "Chitta", "all": [], "any": ["chitta", "a-register", "adangal", "village account"]},
    {"label": "Encumbrance Certificate", "all": [], "any": ["encumbrance certificate", "ec", "sub-registrar", "search period", "nil encumbrance"], "regex": [r"\bencumbrance\b", r"\bec\b"]},
    {"label": "TSLR", "all": [], "any": ["tslr", "town survey land register", "town survey", "ts no", "ward", "block"]},
    {"label": "FMB Sketch", "all": [], "any": ["fmb", "field measurement book", "sub division", "survey sketch", "field sketch"]},
    {"label": "Layout Approval", "all": [], "any": ["layout approval", "dtcp", "cmda", "approved layout", "planning permit"]},
    {"label": "Sale Deed", "all": [], "any": ["sale deed", "deed of sale", "absolute sale", "vendor", "vendee", "schedule property", "consideration", "conveyance", "purchaser", "seller"]},
    {"label": "Settlement Deed", "all": [], "any": ["settlement deed", "settlor", "beneficiary", "family settlement"]},
    {"label": "Partition Deed", "all": [], "any": ["partition deed", "partition", "co-parceners", "coparceners", "share allotment"]},
    {"label": "Release Deed", "all": [], "any": ["release deed", "relinquishment deed", "release", "relinquish"]},
    {"label": "Rectification Deed", "all": [], "any": ["rectification deed", "correction deed", "rectify", "typographical error"]},
    {"label": "Rental Agreement", "all": [], "any": ["rental agreement", "lease agreement", "lease deed", "landlord", "tenant", "lessor", "lessee", "monthly rent", "security deposit", "premises"]},
    {"label": "Lease Deed", "all": [], "any": ["lease deed", "lessor", "lessee", "rent", "lease term"]},
    {"label": "Power of Attorney", "all": [], "any": ["power of attorney", "principal", "agent", "poa", "constituted attorney"]},
    {"label": "Mortgage Deed", "all": [], "any": ["mortgage deed", "mortgagor", "mortgagee", "equitable mortgage", "loan secured", "hypothecation"]},
    {"label": "Affidavit", "all": ["affidavit"], "any": ["deponent", "solemnly", "affirm", "sworn"]},
    {"label": "Employment Contract", "all": [], "any": ["employment contract", "employee", "employer", "salary", "position", "probation", "notice period"]},
    {"label": "Partnership Deed", "all": [], "any": ["partnership deed", "partners", "firm", "profit sharing", "partnership"]},
    {"label": "Corporate Agreement", "all": [], "any": ["master service agreement", "msa", "shareholders agreement", "board", "company", "nda", "service level agreement", "sla"]},
    {"label": "MOA / AOA", "all": [], "any": ["memorandum of association", "moa", "articles of association", "aoa", "company incorporation"]},
    {"label": "Regulatory Filing", "all": [], "any": ["regulatory filing", "roc filing", "mca", "form 32", "form 8", "form 23", "form mgt-7", "form aoc-4", "cin", "din"], "regex": [r"\b[LU]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6}\b", r"\bDIN\s*[:\-]?\s*\d{8}\b"]},
    {"label": "Native Certificate", "all": ["native", "certificate"], "any": ["nativity"]},
    {"label": "Insurance Document", "all": ["insurance"], "any": ["policy", "premium", "insurer", "insured"]},
    {"label": "Gift Deed", "all": [], "any": ["gift deed", "deed of gift", "donor", "donee"]},
    {"label": "Will", "all": ["will"], "any": ["testator", "bequeath", "legacy", "testament"]},
    {"label": "Court Order", "all": [], "any": ["court order", "ordered that", "it is hereby ordered", "interim order", "final order"]},
    {"label": "Court Judgment", "all": [], "any": ["judgment", "judgement", "petitioner", "respondent", "pronounced on", "high court", "supreme court"]},
    {"label": "Legal Notice", "all": [], "any": ["legal notice", "hereby called upon", "notice under", "cause of action", "failing which"]},
    {"label": "Petition", "all": [], "any": ["petition", "writ petition", "civil petition", "criminal petition", "petitioner", "respondent"]},
    {"label": "e-Aadhaar", "all": ["aadhaar"], "any": ["e-aadhaar", "eaadhaar", "e aadhaar", "unique identification authority"]},
    {"label": "Aadhaar PVC Card", "all": ["aadhaar", "pvc"], "any": []},
    {"label": "Aadhaar Card", "all": ["aadhaar"], "any": []},
    {"label": "e-PAN", "all": ["pan"], "any": ["e-pan", "epan", "income tax department"]},
    {"label": "PAN Card", "all": ["pan"], "any": ["income tax", "permanent account number"], "regex": [r"\b[A-Z]{5}\d{4}[A-Z]\b"]},
    {"label": "e-Voter ID", "all": ["voter"], "any": ["e-epic", "e voter", "e-voter", "epic"]},
    {"label": "Voter ID", "all": ["voter"], "any": ["epic", "election commission"], "regex": [r"\b[A-Z]{3}\d{7}\b"]},
    {"label": "e-Driving Licence", "all": ["driving", "licence"], "any": ["e-driving", "e driving", "digilocker"]},
    {"label": "Driving Licence", "all": ["driving", "licence"], "any": ["rto", "dl no"], "regex": [r"\b[A-Z]{2}\s?\d{2}\s?\d{4}\s?\d{7}\b"]},
    {"label": "Learner Licence", "all": ["learner", "licence"], "any": ["ll no", "learning"]},
    {"label": "Vehicle RC", "all": ["registration", "certificate"], "any": ["rc", "chassis", "vehicle"], "regex": [r"\b[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}\b", r"\b(chassis|chassis no\.?|vin)\s*[:\-]?\s*[A-HJ-NPR-Z0-9]{17}\b"]},
    {"label": "PUC Certificate", "all": ["pollution", "certificate"], "any": ["puc"]},
    {"label": "Motor Insurance", "all": ["insurance"], "any": ["motor", "vehicle", "policy"], "regex": [r"\bpolicy\s*no\.?\s*[:\-]?\s*[A-Z0-9\-]{8,20}\b"]},
    {"label": "Passport", "all": ["passport"], "any": ["nationality", "republic of india"], "regex": [r"\b[A-Z]\d{7}\b", r"\bP<[A-Z]{3}[A-Z<]{39}\b", r"\b[0-9]{7}[A-Z0-9<]{1}\b"]},
    {"label": "OCI Card", "all": ["oci"], "any": ["overseas citizen"]},
    {"label": "PIO Card", "all": ["pio"], "any": ["person of indian origin"]},
    {"label": "Birth Certificate", "all": ["birth", "certificate"], "any": ["date of birth"]},
    {"label": "Death Certificate", "all": ["death", "certificate"], "any": ["date of death"]},
    {"label": "Marriage Certificate", "all": ["marriage", "certificate"], "any": ["registration"]},
    {"label": "Divorce Decree", "all": ["divorce"], "any": ["decree", "family court"]},
    {"label": "Adoption Certificate", "all": ["adoption", "certificate"], "any": ["adoptive"]},
    {"label": "Caste Certificate", "all": ["caste", "certificate"], "any": []},
    {"label": "Income Certificate", "all": ["income", "certificate"], "any": ["annual", "revenue"]},
    {"label": "Domicile Certificate", "all": ["domicile", "certificate"], "any": []},
    {"label": "Residence Certificate", "all": ["residence", "certificate"], "any": ["address"]},
    {"label": "EWS Certificate", "all": ["ews", "certificate"], "any": ["economically weaker"]},
    {"label": "Non-Creamy Layer Certificate", "all": ["non", "creamy", "layer"], "any": ["obc"]},
    {"label": "Disability Certificate", "all": ["disability", "certificate"], "any": ["percentage"]},
    {"label": "Senior Citizen Certificate", "all": ["senior", "citizen"], "any": ["age"]},
    {"label": "Ration Card", "all": ["ration", "card"], "any": ["fps", "food"]},
    {"label": "e-Ration Card", "all": ["ration", "card"], "any": ["e-ration", "digital"]},
    {"label": "Legal Heir Certificate", "all": ["legal", "heir"], "any": ["deceased"]},
    {"label": "Transfer Certificate", "all": ["transfer", "certificate"], "any": ["school", "tc"]},
    {"label": "Bonafide Certificate", "all": ["bonafide", "certificate"], "any": ["student", "institution"]},
    {"label": "Migration Certificate", "all": ["migration", "certificate"], "any": ["board", "university"]},
    {"label": "Employee ID", "all": ["employee", "id"], "any": ["company", "department"]},
    {"label": "Experience Certificate", "all": ["experience", "certificate"], "any": ["employment"]},
    {"label": "Salary Certificate", "all": ["salary", "certificate"], "any": ["employer"]},
    {"label": "Labour Card", "all": ["labour", "card"], "any": ["worker"]},
    {"label": "NREGA Job Card", "all": ["nrega", "job", "card"], "any": ["mgnrega"]},
    {"label": "ESIC Card", "all": ["esic", "card"], "any": ["insurance"], "regex": [r"\b\d{10}\b"]},
    {"label": "GST Certificate", "all": ["gst", "certificate"], "any": ["gstin"], "regex": [r"\b\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b"]},
    {"label": "Trade License", "all": ["trade", "license"], "any": ["municipal"]},
    {"label": "Professional Tax Certificate", "all": ["professional", "tax"], "any": ["pt"]},
    {"label": "Farmer ID", "all": ["farmer", "id"], "any": ["agriculture"]},
    {"label": "Kisan Credit Card", "all": ["kisan", "credit", "card"], "any": ["kcc"]},
    {"label": "Soil Health Card", "all": ["soil", "health", "card"], "any": ["test"]},
    {"label": "Ayushman Card", "all": ["ayushman", "card"], "any": ["pm-jay", "beneficiary"]},
    {"label": "Medical Fitness Certificate", "all": ["medical", "fitness", "certificate"], "any": ["doctor"]},
    {"label": "Bank Passbook", "all": ["passbook"], "any": ["account", "ifsc"], "regex": [r"\b[A-Z]{4}0[A-Z0-9]{6}\b"]},
    {"label": "Bank Statement", "all": ["bank", "statement"], "any": ["transaction"], "regex": [r"\b[A-Z]{4}0[A-Z0-9]{6}\b"]},
    {"label": "Cancelled Cheque", "all": ["cheque"], "any": ["cancelled", "canceled"]},
    {"label": "Debit Card", "all": ["debit", "card"], "any": ["bank"]},
    {"label": "Credit Card Statement", "all": ["credit", "card", "statement"], "any": ["outstanding"]},
    # --- General document types (added for broader coverage) ---
    {"label": "Resume", "all": [], "any": ["resume", "curriculum vitae", "cv", "education", "experience", "skills"], "regex": [r"\b(Education|Experience|Skills|Projects|Summary)\b"]},
    {"label": "Invoice", "all": [], "any": ["invoice", "tax invoice", "gst", "gstin", "bill to", "invoice number", "subtotal", "total", "amount due"], "regex": [r"\b(invoice\s*(no|number)\s*[:\-]?\s*[A-Za-z0-9\-]+)\b", r"\bGSTIN\b"]},
    {"label": "ID Card", "all": [], "any": ["identity card", "id card", "id no", "date of birth", "govt of", "government of"], "regex": [r"(?i)\b(id\s*(no|number)\s*[:\-]?\s*[A-Za-z0-9\-]+)\b"]},
    {"label": "Contract", "all": [], "any": ["contract", "contract agreement"], "regex": [r"\b(contract)\b"]},
    {"label": "Affidavit", "all": [], "any": ["affidavit", "sworn statement", "deponent", "notary"], "regex": [r"\baffidavit\b"]},
    {"label": "Legal Agreement", "all": [], "any": ["agreement", "between", "parties", "witnesseth"], "regex": [r"\bagreement\b", r"\bbetween\b"]},
    {"label": "Legal Document", "all": [], "any": ["hereby", "witnesseth", "whereas", "parties", "agreement", "deed", "dated", "between", "signed", "executed"]},
]

_DOC_TYPE_RULES_COMPILED = []
for r in DOC_TYPE_RULES:
    compiled = []
    for pat in r.get("regex", []):
        try:
            compiled.append(re.compile(pat))
        except Exception:
            continue
    _DOC_TYPE_RULES_COMPILED.append({**r, "regex_compiled": compiled})

class DocumentClassifier:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.pipeline = None
        self.is_loaded = False
        self._load_model()

    def _load_model(self):
        try:
            if os.path.exists(ALT_PIPELINE_PATH):
                warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
                self.pipeline = joblib.load(ALT_PIPELINE_PATH)
                self.model = getattr(self.pipeline, "named_steps", {}).get("clf", None)
                self.vectorizer = getattr(self.pipeline, "named_steps", {}).get("tfidf", None)
                self.is_loaded = self._validate_loaded_model()
                if self.is_loaded:
                    print(f"ML Pipeline loaded from {ALT_PIPELINE_PATH}")
                else:
                    print(f"ML Pipeline at {ALT_PIPELINE_PATH} is incompatible/unfitted. Will fallback.")
            elif os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
                warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
                self.model = joblib.load(MODEL_PATH)
                self.vectorizer = joblib.load(VECTORIZER_PATH)
                self.is_loaded = self._validate_loaded_model()
                if self.is_loaded:
                    print(f"ML Model loaded from {MODEL_PATH}")
                else:
                    print(f"ML Model at {MODEL_PATH} is incompatible/unfitted. Will fallback.")
            else:
                print(f"ML Model not found at {MODEL_PATH}")
        except Exception as e:
            print(f"Failed to load ML model: {e}")

    def _validate_loaded_model(self) -> bool:
        try:
            sample = "This legal agreement between parties includes liability and termination clauses."
            if self.pipeline is not None:
                _ = self.pipeline.predict([sample])[0]
                return True
            if self.model is None or self.vectorizer is None:
                return False
            X = self.vectorizer.transform([sample])
            _ = self.model.predict(X)[0]
            return True
        except Exception:
            self.pipeline = None
            self.model = None
            self.vectorizer = None
            return False

    def get_supported_document_types(self) -> List[str]:
        labels: List[str] = []
        for rule in DOC_TYPE_RULES:
            label = (rule or {}).get("label")
            if label and label not in labels:
                labels.append(label)
        return labels

    def train_model(self, texts: List[str], labels: List[str]) -> Dict[str, Any]:
        try:
            if not texts or not labels:
                return {"success": False, "error": "No training data provided"}

            print(f"Training model with {len(texts)} samples...")
            # Create a pipeline with TF-IDF and SVM (SGDClassifier)
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words='english')
            self.model = SGDClassifier(loss='modified_huber', penalty='l2', alpha=1e-4, random_state=42, max_iter=1000, tol=1e-3)
            
            X_train = self.vectorizer.fit_transform(texts)
            self.model.fit(X_train, labels)

            os.makedirs(MODEL_DIR, exist_ok=True)
            
            joblib.dump(self.model, MODEL_PATH)
            joblib.dump(self.vectorizer, VECTORIZER_PATH)
            
            self.is_loaded = True
            print(f"Model saved to {MODEL_PATH}")
            return {"success": True, "details": f"Trained on {len(texts)} samples."}

        except Exception as e:
            print(f"Training failed: {e}")
            return {"success": False, "error": str(e)}

    def _iter_samples_from_file(self, f, fmt: str) -> Iterable[Tuple[str, str]]:
        if fmt == "csv":
            rdr = csv.DictReader(f)
            for row in rdr:
                t = (row or {}).get("text") or ""
                l = (row or {}).get("label") or ""
                t = t.strip()
                l = l.strip()
                if t and l:
                    yield (t, l)
        elif fmt == "ndjson":
            for line in f:
                line = (line or "").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                t = (obj or {}).get("text") or ""
                l = (obj or {}).get("label") or ""
                t = t.strip()
                l = l.strip()
                if t and l:
                    yield (t, l)
        elif fmt == "json_array":
            try:
                arr = json.load(f)
            except Exception:
                arr = []
            if isinstance(arr, list):
                for it in arr:
                    t = (it or {}).get("text") or ""
                    l = (it or {}).get("label") or ""
                    t = t.strip()
                    l = l.strip()
                    if t and l:
                        yield (t, l)
        else:
            return

    def train_model_incremental_from_path(
        self,
        path: str,
        fmt: str,
        batch_size: int = 1000,
        job_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
        save_pipeline_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Incremental training for large datasets.
        fmt: csv | ndjson | json_array
        """
        try:
            if not os.path.exists(path):
                return {"success": False, "error": f"Dataset not found: {path}"}

            # First pass: collect labels + count
            labels_set = set()
            total = 0
            with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
                for _, label in self._iter_samples_from_file(f, fmt):
                    labels_set.add(label)
                    total += 1
                    if job_hook and total % max(1000, batch_size) == 0:
                        job_hook({"phase": "scan", "processed": total})

            if not labels_set:
                return {"success": False, "error": "No training data provided"}

            classes = np.array(sorted(labels_set))

            # Incremental pipeline (fixed memory)
            self.vectorizer = HashingVectorizer(
                n_features=2**20,
                alternate_sign=False,
                norm="l2",
                ngram_range=(1, 2),
                stop_words="english",
            )
            self.model = SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=1e-4,
                random_state=42,
            )

            processed = 0
            batch_texts: List[str] = []
            batch_labels: List[str] = []

            with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
                for text, label in self._iter_samples_from_file(f, fmt):
                    batch_texts.append(text)
                    batch_labels.append(label)
                    if len(batch_texts) >= batch_size:
                        X = self.vectorizer.transform(batch_texts)
                        self.model.partial_fit(X, batch_labels, classes=classes)
                        processed += len(batch_texts)
                        batch_texts = []
                        batch_labels = []
                        if job_hook:
                            job_hook({"phase": "train", "processed": processed, "total": total})

            if batch_texts:
                X = self.vectorizer.transform(batch_texts)
                self.model.partial_fit(X, batch_labels, classes=classes)
                processed += len(batch_texts)
                if job_hook:
                    job_hook({"phase": "train", "processed": processed, "total": total})

            os.makedirs(MODEL_DIR, exist_ok=True)
            joblib.dump(self.model, MODEL_PATH)
            joblib.dump(self.vectorizer, VECTORIZER_PATH)

            if save_pipeline_path:
                os.makedirs(os.path.dirname(save_pipeline_path), exist_ok=True)
                pipe = Pipeline([("tfidf", self.vectorizer), ("clf", self.model)])
                joblib.dump(pipe, save_pipeline_path)

            self.is_loaded = True
            return {
                "success": True,
                "details": f"Incremental trained on {processed} samples.",
                "total": total,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def detect_document_type_by_rules(self, text: str) -> Optional[str]:
        t = (text or "").lower()
        if not t.strip():
            return None
        print(f"[DEBUG] detect_document_type_by_rules: text_len={len(t)}")
        for rule in _DOC_TYPE_RULES_COMPILED:
            all_ok = all(k in t for k in rule.get("all", []))
            any_list = rule.get("any", [])
            any_ok = True if not any_list else any(k in t for k in any_list)
            regex_list = rule.get("regex_compiled", [])
            regex_ok = True if not regex_list else any(rx.search(text or "") for rx in regex_list)
            if all_ok and any_ok and regex_ok:
                print(f"[DEBUG] detect_document_type_by_rules: Matched '{rule['label']}'")
                return rule["label"]
        print("[DEBUG] detect_document_type_by_rules: No rule match found")
        return None

    def predict_document_type(self, text: str) -> Tuple[str, float]:
        if text is None or not text.strip():
            print(f"[DEBUG] ML Service: Empty text.")
            return "Generic Legal Document", 0.0
        if not self.is_loaded:
            print(f"[DEBUG] ML Service: Model not loaded.")
            return "Unknown", 0.0

        try:
            clean_text = re.sub(r'\s+', ' ', text).strip()
            # Handle short text
            if len(clean_text) < 10:
                print(f"[DEBUG] ML Service: Text too short ({len(clean_text)})")
                return "Unknown", 0.0
            if self.pipeline is not None:
                prediction = self.pipeline.predict([clean_text])[0]
                probabilities = self.pipeline.predict_proba([clean_text])[0]
            else:
                X_test = self.vectorizer.transform([clean_text])
                prediction = self.model.predict(X_test)[0]
                probabilities = self.model.predict_proba(X_test)[0]

            confidence = float(np.max(probabilities))
            print(f"[DEBUG] ML Service: Prediction='{prediction}', Confidence={confidence:.4f}, Probabilities={probabilities}")

            if confidence < 0.50:
                return "Other Document", confidence
            return prediction, confidence
        except Exception as e:
            print(f"[DEBUG] ML Service: Prediction error: {e}")
            return "Generic Legal Document", 0.0

    def verify_document_ml(self, text: str) -> Dict[str, Any]:
        """
        Verify document legitimacy using ML confidence and heuristic rules.
        """
        safe_text = text or ""
        safe_text_lower = safe_text.lower()

        if not safe_text.strip():
            return {
                "marker": "Rejected",
                "ai_confidence": 0,
                "risk_level": "Unknown",
                "issues": ["Empty text", "Model could not classify document (Confidence: 0%)."],
                "predicted_category": "Unknown"
            }

        cat, conf = self.predict_document_type(safe_text)
        rule_cat = self.detect_document_type_by_rules(safe_text)

        print(f"[DEBUG] verify_document_ml: Initial Cat='{cat}', Conf={conf:.4f}, RuleCat='{rule_cat}'")

        # 1. DOCUMENT TYPE MATCHING WITH LEGAL INDICATOR GUARD
        legal_indicators = [
            "agreement","deed","affidavit","court","notary","jurisdiction",
            "petitioner","respondent","stamp duty","witness"
        ]
        indicator_count = sum(1 for k in legal_indicators if k in safe_text_lower)
        if rule_cat == "Resume" or cat == "Resume":
            cat = "Resume"
            conf = max(conf, 0.90)
        elif rule_cat:
            if conf >= 0.75 and indicator_count >= 1:
                print(f"[DEBUG] verify_document_ml: Applying guarded rule override. '{cat}' -> '{rule_cat}'")
                cat = rule_cat
                conf = max(conf, 0.92)
            else:
                print(f"[DEBUG] verify_document_ml: Skip override due to low confidence or missing legal indicators.")
        elif cat in ["Generic Legal Document", "Unknown", "Other Document"] and conf < 0.70:
            conf = max(conf, 0.40)

        ai_confidence = int(conf * 100)

        # 2. MARKER & RISK CALCULATION
        marker = "Verified"
        risk_level = "Low"
        issues = []

        if ai_confidence < 45:
            marker = "Rejected"
            risk_level = "Critical"
            issues.append(f"Model could not classify document (Confidence: {ai_confidence}%).")
        elif ai_confidence < 75:
            marker = "Flagged"
            risk_level = "High"
            issues.append(f"Low classification confidence ({ai_confidence}%) for category '{cat}'")
        elif ai_confidence < 88:
            risk_level = "Medium"

        print(f"[DEBUG] verify_document_ml: AI Confidence={ai_confidence}%, Marker='{marker}', Risk='{risk_level}'")

        # 3. HEURISTICS
        # Text Length
        text_len = len(safe_text.strip())
        if text_len < 150:
            if rule_cat:
                if risk_level == "Low":
                    risk_level = "Medium"
                issues.append(f"Document text is relatively short ({text_len} chars), but matched via rules.")
            else:
                marker = "Rejected"
                risk_level = "Critical"
                issues.append(f"Document text is too short ({text_len} chars) for reliable analysis.")
                ai_confidence = 0

        # Structure Check
        legal_keywords = ["agreement", "deed", "parties", "witness", "signed", "dated", "hereby", "schedule", "property"]
        keyword_count = sum(1 for k in legal_keywords if k in safe_text_lower)
        
        if "agreement" in cat.lower() or "deed" in cat.lower() or cat == "Sale Deed":
            if keyword_count < 3:
                risk_level = "High" if risk_level != "Critical" else "Critical"
                issues.append("Missing standard legal structure keywords.")
                marker = "Flagged"
                ai_confidence = max(25, ai_confidence - 20)
            elif keyword_count >= 6:
                # Strong legal structure found
                ai_confidence = min(100, ai_confidence + 10)

        # Non-legal safety: low confidence with no legal indicators -> Non-Legal Document
        if ai_confidence < 50 and indicator_count == 0 and cat not in {"Resume","Invoice","ID Card"}:
            cat = "Non-Legal Document"
            marker = "Verified"
            risk_level = "None"
            ai_confidence = 85

        # Ensure we don't return exactly 25% too often unless it's a real failure
        if marker == "Verified" and ai_confidence < 50:
            ai_confidence = 70 # Floor for verified docs

        return {
            "marker": marker,
            "ai_confidence": ai_confidence,
            "risk_level": risk_level,
            "issues": issues,
            "predicted_category": cat
        }

    def evaluate_dataset(self, texts: List[str], labels: List[str]) -> Dict[str, Any]:
        try:
            if not texts or not labels:
                return {"success": False, "error": "No evaluation data provided"}
            if not self.is_loaded:
                return {"success": False, "error": "Model not loaded"}
            X = self.vectorizer.transform(texts)
            preds = self.model.predict(X)
            acc = float(accuracy_score(labels, preds))
            report = classification_report(labels, preds, output_dict=True, zero_division=0)
            cm = confusion_matrix(labels, preds).tolist()
            return {"success": True, "accuracy": acc, "report": report, "confusion_matrix": cm}
        except Exception as e:
            return {"success": False, "error": str(e)}

classifier = DocumentClassifier()
