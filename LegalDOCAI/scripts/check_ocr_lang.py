import os, sys
project_root = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.normpath(os.path.join(project_root, ".."))
if repo_root not in sys.path:
    sys.path.append(repo_root)
from backend.app.services.ocr_service import is_tesseract_language_available, _get_pytesseract

def main():
    try:
        pt = _get_pytesseract()
        langs = pt.get_languages(config="") if pt else []
        print("pytesseract.get_languages:", langs)
    except Exception as e:
        print("pytesseract.get_languages error:", e)
    for code in ["eng", "tam", "hin", "tel"]:
        try:
            print(code, "available:", is_tesseract_language_available(code))
        except Exception as e:
            print(code, "check error:", e)

if __name__ == "__main__":
    main()
