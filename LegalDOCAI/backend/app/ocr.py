import os
import pytesseract
from pdf2image import convert_from_path

_tess_cmd = os.getenv("TESSERACT_CMD")
if _tess_cmd:
    pytesseract.pytesseract.tesseract_cmd = _tess_cmd

def extract_text_from_file(filepath: str) -> str:
    """Extract text from a PDF file using OCR."""
    images = convert_from_path(filepath)
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img)
    return text
