import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.ocr import extract_text_from_image, extract_text_from_pdf
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

print("Image OCR:")
print(extract_text_from_image(r"D:\Project\document2.png"))

print("PDF OCR:")
print(extract_text_from_pdf(r"D:\Project\document1.pdf"))
