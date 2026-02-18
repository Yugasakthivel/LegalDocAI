
import sys
import os

print("Testing imports...")
try:
    import fastapi
    print("fastapi ok")
    import uvicorn
    print("uvicorn ok")
    import spacy
    print("spacy ok")
    from backend.app.services.ml_service import classifier
    print("ml_service ok")
    from backend.app.routes import router
    print("routes ok")
    print("All imports successful!")
except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()
