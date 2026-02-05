import sys
import os
import json

# Add project root to sys.path so we can import app modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from backend.app.services.ml_service import classifier

DATA_FILE = os.path.join(project_root, "legal_dataset/training_data.json")

def main():
    print("🚀 Starting Model Training Script")
    
    if not os.path.exists(DATA_FILE):
        print(f"❌ Data file not found at {DATA_FILE}")
        print("Please run generate_synthetic_data.py first.")
        return

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        texts = [d["text"] for d in data]
        labels = [d["category"] for d in data]
        
        print(f"Loaded {len(texts)} samples.")
        
        result = classifier.train_model(texts, labels)
        print(result)
        
    except Exception as e:
        print(f"❌ Error during training: {e}")

if __name__ == "__main__":
    main()
