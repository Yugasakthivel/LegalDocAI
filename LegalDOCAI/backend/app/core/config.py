# ================================
# config.py — LegalDocAI Settings
# Author: Sakthi K
# ================================

import os
from dotenv import load_dotenv
import pytesseract

# ---------------- Load .env file ----------------
# Adjusted path to find .env in root or local
load_dotenv()

# ---------------- Security Configuration ----------------
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-for-jwt-generation-keep-it-safe")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))

# ---------------- OpenAI Configuration ----------------
# REMOVED: OpenAI dependency removed in favor of local ML.
OPENAI_API_KEY = None
OPENAI_MODEL = "disabled"
SUMMARIZATION_MODEL = os.getenv("SUMMARIZATION_MODEL", "sshleifer/distilbart-cnn-12-6")
EMBEDDING_MODEL = "disabled"

# ---------------- MongoDB Configuration ----------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "LegalDocAI_DB")

# ---------------- Tesseract OCR Configuration ----------------
TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# ---------------- File Upload Configuration ----------------
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER") or os.getenv("UPLOAD_DIR") or "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize OpenAI client (Lazy load preferred but keeping current pattern)
client = None


# ---------------- Optional Startup Logs ----------------
def print_config():
    print("===============================================")
    print("LegalDocAI Configuration Loaded Successfully")
    print("OpenAI Model:", OPENAI_MODEL)
    print("MongoDB URI:", MONGO_URI)
    print("Database Name:", DB_NAME)
    print(f"Tesseract Path: {TESSERACT_CMD if TESSERACT_CMD else 'Not configured'}")
    print("===============================================")

print_config()
