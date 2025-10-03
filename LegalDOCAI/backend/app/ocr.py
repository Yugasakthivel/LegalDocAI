import pytesseract
from pdf2image import convert_from_path

# Tell pytesseract where the exe is installed
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_from_file(filepath: str) -> str:
    """Extract text from a PDF file using OCR."""
    images = convert_from_path(filepath)
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img)
    return text
