import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Settings:
    APP_NAME: str = "LegalDocAI Backend Skeleton"
    VERSION: str = "0.1.0"
    
    UPLOAD_FOLDER: str = os.getenv("UPLOAD_FOLDER", "uploads")
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
    ALLOWED_EXTENSIONS: set = set(
        os.getenv("ALLOWED_EXTENSIONS", "pdf,jpg,jpeg,png,docx,txt").split(",")
    )
    
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    TESSERACT_CMD: str = os.getenv("TESSERACT_CMD", "")
    
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    MONGODB_DB_NAME: str = os.getenv("MONGODB_DB_NAME", "legaldocai_db")
    
    @property
    def max_upload_size_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    
    def ensure_upload_folder(self):
        Path(self.UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.ensure_upload_folder()
