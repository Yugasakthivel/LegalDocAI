import os, json, joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../LegalDOCAI/backend
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_PATH = os.path.join(DATA_DIR, "training_data.json")
PIPELINE_PATH = os.path.join(MODELS_DIR, "document_model.pkl")

def train_from_dataset():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [d.get("text","") for d in data if d.get("text")]
    labels = [d.get("label","") for d in data if d.get("label")]
    if not texts or not labels:
        raise ValueError("Dataset invalid: missing text/label")
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=10000)),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    pipe.fit(texts, labels)
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(pipe, PIPELINE_PATH)
    return {"message": "Model trained successfully", "samples": len(texts)}

if __name__ == "__main__":
    print(train_from_dataset())
