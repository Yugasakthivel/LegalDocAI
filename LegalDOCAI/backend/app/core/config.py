# ================================
# config.py — LegalDocAI Settings
# Author: Sakthi K
# ================================

import os
from dotenv import load_dotenv

# ---------------- Load .env file ----------------
# Adjusted path to find .env in root or local
load_dotenv()

# ---------------- Security Configuration ----------------
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))
if not SECRET_KEY or SECRET_KEY.strip() == "":
    import secrets
    SECRET_KEY = secrets.token_hex(32)
    try:
        print("SECURITY INFO: Generated ephemeral SECRET_KEY for runtime")
    except Exception:
        pass

# ---------------- OpenAI Configuration ----------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SUMMARIZATION_MODEL = os.getenv("SUMMARIZATION_MODEL", "sshleifer/distilbart-cnn-12-6")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
PIPELINE_TIMEOUT_MS = int(os.getenv("PIPELINE_TIMEOUT_MS", "20000"))
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "25"))
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "20"))

RAG_MODE = os.getenv("RAG_MODE", "auto")
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/paraphrase-MiniLM-L3-v2")
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "300"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
RAG_MIN_CONFIDENCE = float(os.getenv("RAG_MIN_CONFIDENCE", "0.3"))
# ---------------- MongoDB Configuration ----------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "LegalDocAI_DB")

# ---------------- Tesseract OCR Configuration ----------------
TESSERACT_CMD = os.getenv("TESSERACT_CMD")

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


