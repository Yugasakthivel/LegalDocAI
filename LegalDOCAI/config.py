# ================================
# config.py — LegalDocAI Settings
# Author: Sakthi K
# ================================

import os
from dotenv import load_dotenv
from openai import OpenAI  # Official OpenAI SDK (v1+)
import pytesseract

# ---------------- Load .env file ----------------
load_dotenv()

# ---------------- OpenAI Configuration ----------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("sk-your"):
    raise ValueError(
        "⚠️ OPENAI_API_KEY not set or still placeholder! "
        "Update your .env with a real key from https://platform.openai.com/account/api-keys"
    )

# Initialize OpenAI client
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
    # 🔹 Optional quick verification (lightweight)
    # Comment this out if you want faster startup, or keep for debugging
    # test_response = client.models.list()
except Exception as e:
    raise RuntimeError(f"⚠️ Failed to initialize OpenAI client or verify key: {e}")

# ---------------- MongoDB Configuration ----------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "LegalDocAI_DB")

# ---------------- Tesseract OCR Configuration ----------------
TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# ---------------- Optional Startup Logs ----------------
def print_config():
    print("===============================================")
    print("🚀 LegalDocAI Configuration Loaded Successfully")
    print("✅ OpenAI Model:", OPENAI_MODEL)
    print("✅ MongoDB URI:", MONGO_URI)
    print("✅ Database Name:", DB_NAME)
    print(f"✅ Tesseract Path: {TESSERACT_CMD if TESSERACT_CMD else 'Not configured'}")
    print("===============================================")

print_config()
