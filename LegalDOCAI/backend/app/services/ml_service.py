import os
import joblib
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
7→from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from typing import Dict, Any, List, Tuple

# Constants
# Path: backend/app/services/ml_service.py
# Goal: Reach LegalDOCAI/data/models
# From services: .. (app) -> .. (backend) -> .. (LegalDOCAI) -> data/models
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODEL_DIR = os.path.join(BASE_DIR, "data", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "doc_classifier.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")

class DocumentClassifier:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.is_loaded = False
        self._load_model()

    def _load_model(self):
        try:
            if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
                self.model = joblib.load(MODEL_PATH)
                self.vectorizer = joblib.load(VECTORIZER_PATH)
                self.is_loaded = True
                print(f"ML Model loaded from {MODEL_PATH}")
            else:
                print(f"ML Model not found at {MODEL_PATH}")
        except Exception as e:
            print(f"Failed to load ML model: {e}")

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

    def predict_document_type(self, text: str) -> Tuple[str, float]:
        if not self.is_loaded or not text.strip():
            return "Generic Legal Document", 0.0

        try:
            clean_text = re.sub(r'\s+', ' ', text).strip()
            # Handle short text
            if len(clean_text) < 10:
                return "Unknown", 0.0
                
            X_test = self.vectorizer.transform([clean_text])
            prediction = self.model.predict(X_test)[0]
            probabilities = self.model.predict_proba(X_test)[0]
            confidence = float(np.max(probabilities))
            return prediction, confidence
        except Exception as e:
            print(f"Prediction error: {e}")
            return "Generic Legal Document", 0.0

82→    def evaluate_dataset(self, texts: List[str], labels: List[str]) -> Dict[str, Any]:
83→        try:
84→            if not self.is_loaded:
85→                return {"success": False, "error": "Model not loaded"}
86→            if not texts or not labels:
87→                return {"success": False, "error": "No evaluation data provided"}
88→            X = self.vectorizer.transform(texts)
89→            preds = self.model.predict(X)
90→            acc = float(accuracy_score(labels, preds))
91→            report = classification_report(labels, preds, output_dict=True, zero_division=0)
92→            cm = confusion_matrix(labels, preds).tolist()
93→            return {"success": True, "accuracy": acc, "report": report, "confusion_matrix": cm}
94→        except Exception as e:
95→            return {"success": False, "error": str(e)}
96→
    def verify_document_ml(self, text: str) -> Dict[str, Any]:
        """
        Verify document legitimacy using ML confidence and heuristic rules.
        """
        cat, conf = self.predict_document_type(text)
        ai_confidence = int(conf * 100)
        
        marker = "Verified"
        risk_level = "Low"
        issues = []

        # 1. Confidence Check
        if ai_confidence < 40:
            marker = "Rejected"
            risk_level = "Critical"
            issues.append(f"Model could not classify document (Confidence: {ai_confidence}%).")
        elif ai_confidence < 70:
            marker = "Flagged"
            risk_level = "High"
            issues.append(f"Low classification confidence ({ai_confidence}%) for category '{cat}'")
        elif ai_confidence < 85:
            risk_level = "Medium"
        
        # 2. Heuristic: Text Length
        if len(text.strip()) < 100:
             marker = "Rejected"
             risk_level = "Critical"
             issues.append("Document text is too short (< 100 chars).")
             ai_confidence = 0
        
        # 3. Heuristic: Structure Check (Keywords)
        # If it claims to be a deed/agreement, it usually has 'signed' or 'witness'
        structure_keywords = ["signed", "witness", "party", "executed", "agreement", "deed"]
        if "agreement" in cat.lower() or "deed" in cat.lower():
            if not any(k in text.lower() for k in structure_keywords):
                risk_level = "High" if risk_level != "Critical" else "Critical"
                issues.append("Missing standard legal keywords (signed, witness, etc).")
                marker = "Flagged"

        return {
            "marker": marker,
            "ai_confidence": ai_confidence,
            "risk_level": risk_level,
            "issues": issues,
            "predicted_category": cat
        }

classifier = DocumentClassifier()
