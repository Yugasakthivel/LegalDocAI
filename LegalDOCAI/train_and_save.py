import sys
import os
import json

# Add current directory to sys.path to ensure local imports work
sys.path.append(os.getcwd())

# Import safe from backend (assuming running from LegalDOCAI root)
try:
    from backend.app.services.ml_service import classifier, MODEL_PATH
except ImportError:
    print("Could not import backend. Make sure you are running from the LegalDOCAI folder.")
    sys.exit(1)

# Path to dataset: LegalDOCAI/../legal_dataset/training_data.json
DATA_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "legal_dataset", "training_data.json"))

def main():
    print("=== Model Training Utility ===")
    print(f"Data File: {DATA_FILE}")
    print(f"Target Model Path: {MODEL_PATH}")

    try:
        if not os.path.exists(DATA_FILE):
            print(f"ERROR: Data file not found at {DATA_FILE}")
            return

        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        texts = [d["text"] for d in data]
        labels = [d["category"] for d in data]
        
        print(f"Loaded {len(texts)} samples. Training...")
        
        result = classifier.train_model(texts, labels)
        print(f"Training Result: {result}")
        
    except Exception as e:
        print(f"FATAL ERROR: {e}")

if __name__ == "__main__":
    main()
