import os
import io
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from typing import Tuple, List
from .config import TESSERACT_CMD

# If user set TESSERACT_CMD in env, configure pytesseract
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> Tuple[str, List[Image.Image]]:
    """
    Extracts textual content and embedded images from a PDF bytes.
    Returns: (extracted_text, list_of_images)
    """
    text_parts = []
    images = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            # text
            page_text = page.get_text("text") or ""
            text_parts.append(page_text)

            # extract images from the page
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                images.append(image)

    all_text = "\n".join(text_parts).strip()
    return all_text, images


def ocr_image(pil_image: Image.Image) -> str:
    """Run pytesseract OCR on a PIL image and return text."""
    text = pytesseract.image_to_string(pil_image)
    return text


def process_uploaded_file_bytes(filename: str, file_bytes: bytes) -> dict:
    """
    Main processing pipeline:
      - If PDF: extract text and images -> OCR images
      - If text-like: decode bytes to text
      - If image file: OCR directly
    Returns metadata dict with extracted_text, ocr_texts, combined_text
    """
    ext = os.path.splitext(filename.lower())[1]
    extracted_text = ""
    ocr_texts = []

    if ext in [".pdf"]:
        text, images = extract_text_from_pdf_bytes(file_bytes)
        extracted_text = text
        # OCR each image
        for im in images:
            t = ocr_image(im)
            if t and t.strip():
                ocr_texts.append(t)
    elif ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"]:
        # direct image OCR
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        t = ocr_image(image)
        ocr_texts.append(t)
    else:
        # trust as text file
        try:
            extracted_text = file_bytes.decode("utf-8")
        except Exception:
            try:
                extracted_text =import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF using PyMuPDF"""
    text = ""
    doc = fitz.open(file_path)
    for page in doc:
        # page.get_text is correct (ignore Pylance warning)
        text += page.get_text("text")
    return text.strip()


def extract_text_from_image(image_bytes: bytes) -> str:
    """Extract text from image using OCR (pytesseract)."""
    image = Image.open(io.BytesIO(image_bytes))
    text = pytesseract.image_to_string(image)
    return text.strip()


def process_uploaded_file_bytes(file_bytes: bytes, filename: str) -> str:
    """Decide if file is PDF or Image, extract text."""
    if filename.lower().endswith(".pdf"):
        with open("temp.pdf", "wb") as f:
            f.write(file_bytes)
        return extract_text_from_pdf("temp.pdf")
    else:
        return extract_text_from_image(file_bytes)
 file_bytes.decode("latin-1")
            except Exception:
                extracted_text = ""

    combined = "\n".join([extracted_text] + ocr_texts).strip()
    return {
        "filename": filename,
        "extracted_text": extracted_text,
        "ocr_texts": ocr_texts,
        "combined_text": combined,
    }
