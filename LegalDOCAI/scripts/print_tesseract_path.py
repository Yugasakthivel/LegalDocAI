import os, sys
repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.append(repo_root)
from backend.app.core.config import TESSERACT_CMD
print("TESSERACT_CMD =", TESSERACT_CMD)
