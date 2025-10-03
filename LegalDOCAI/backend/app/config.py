import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "LegalDocAI_DB")

# Tesseract executable path (only needed on Windows or custom install)
TESSERACT_CMD = os.getenv("TESSERACT_CMD", None)  # e.g., "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Embedding model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Summarization model (optional)
SUMMARIZATION_MODEL = os.getenv("SUMMARIZATION_MODEL", "sshleifer/distilbart-cnn-12-6")
