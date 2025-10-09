import io
from typing import Tuple, List, Dict
from PIL import Image
import pytesseract
import fitz  # PyMuPDF

# Optional: if you set env var to point Tesseract binary, configure here:
# import os
# pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD", pytesseract.pytesseract.tesseract_cmd)


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> Tuple[str, List[Image.Image]]:
    """
    Extracts text (page text) and embedded images from PDF bytes.
    Returns tuple: (concatenated_page_text, list_of_PIL_images)
    """
    text_parts: List[str] = []
    images: List[Image.Image] = []

    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page_index in range(len(doc)):
                page: fitz.Page = doc[page_index]  # type: ignore[attr-defined]
                # page.get_text is valid at runtime — give type hint to help Pylance
                page.get_textbox
                page_text = page.get_textbox("text") or ""
                text_parts.append(page_text)

                # extract images
                for img in page.get_images(full=True):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    img_bytes = base_image.get("image")
                    if img_bytes:
                        try:
                            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                            images.append(pil_img)
                        except Exception:
                            # skip unreadable image
                            continue
    except Exception:
        # If pdf parsing fails, return empty text and no images
        return "", []

    combined_text = "\n".join(text_parts).strip()
    return combined_text, images


def ocr_image(pil_image: Image.Image) -> str:
    """Run pytesseract OCR on a PIL image and return text (str)."""
    try:
        text = pytesseract.image_to_string(pil_image)
        return text.strip()
    except Exception:
        return ""


def process_uploaded_file_bytes(file_bytes: bytes, filename: str) -> Dict[str, object]:
    """
    Unified processing for uploaded bytes.
    Returns a dict with keys:
      - filename (str)
      - extracted_text (str)    # text extracted directly from file (PDF pages or text file)
      - ocr_texts (List[str])   # OCR from images (if any)
      - combined_text (str)     # concatenation of above (used for embeddings/search)
    """
    filename = filename or "uploaded_file"
    ext = (filename or "").lower().rsplit(".", 1)[-1] if "." in filename else ""
    extracted_text = ""
    ocr_texts: List[str] = []

    # PDF handling
    if ext == "pdf":
        extracted_text, images = extract_text_from_pdf_bytes(file_bytes)
        # OCR each extracted image
        for img in images:
            t = ocr_image(img)
            if t:
                ocr_texts.append(t)

    # image handling (common image extensions)
    elif ext in {"png", "jpg", "jpeg", "tiff", "bmp", "gif"}:
        try:
            pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            t = ocr_image(pil_img)
            if t:
                ocr_texts.append(t)
        except Exception:
            extracted_text = ""

    # text-like file (try decode)
    else:
        try:
            extracted_text = file_bytes.decode("utf-8")
        except Exception:
            try:
                extracted_text = file_bytes.decode("latin-1")
            except Exception:
                extracted_text = ""

    combined = "\n".join([p for p in (extracted_text,) + tuple(ocr_texts) if p]).strip()

    return {
        "filename": filename,
        "extracted_text": extracted_text,
        "ocr_texts": ocr_texts,
        "combined_text": combined,
    }
