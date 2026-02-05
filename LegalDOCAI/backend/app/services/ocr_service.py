import os
import re
import shutil
import fitz  # PyMuPDF
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import concurrent.futures
from typing import List, Optional
from fastapi import HTTPException
from backend.app.core.config import TESSERACT_CMD

try:
    import easyocr
except ImportError:
    easyocr = None

try:
    from docx import Document
except ImportError:
    Document = None

try:
    import pandas as pd
except ImportError:
    pd = None

OCR_DPI_DEFAULT = 300

def preprocess_image(img: Image.Image) -> Image.Image:
    """
    Preprocess image to improve OCR accuracy:
    1. Convert to grayscale
    2. Enhance contrast
    3. Enhance sharpness
    4. Binarize (thresholding)
    """
    try:
        # Convert to grayscale
        img = img.convert('L')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.5)
        
        # Apply slight blur to remove noise before thresholding
        # img = img.filter(ImageFilter.MedianFilter(size=3))
        
        # Binarize (Thresholding)
        # This simple thresholding often works well for documents
        img = img.point(lambda x: 0 if x < 140 else 255, '1')
        
        return img
    except Exception:
        return img

def detect_file_type(filename: str) -> str:
    ext = filename.split(".")[-1].lower()
    if ext in ["pdf", "docx", "xls", "xlsx", "png", "jpg", "jpeg", "txt"]:
        return ext
    return "unknown"

def detect_language_simple(text: str) -> str:
    try:
        for ch in text:
            cp = ord(ch)
            if 0x0B80 <= cp <= 0x0BFF:
                return "Tamil"
            if 0x0900 <= cp <= 0x097F:
                return "Hindi"
        return "English" if re.search(r"[A-Za-z]", text) else "Unknown"
    except Exception:
        return "Unknown"

def is_tesseract_language_available(lang: str) -> bool:
    try:
        langs = pytesseract.get_languages(config="")
        return lang in (langs or [])
    except Exception:
        try:
            img = Image.new("RGB", (16, 16), (255, 255, 255))
            _ = pytesseract.image_to_string(img, lang=lang)
            return True
        except Exception:
            return False

def resolve_ocr_lang_for_file(file_path: str, ext: str, requested: str = None) -> str:
    if requested and requested.lower() != "auto":
        return requested
    try:
        tam_avail = is_tesseract_language_available("tam")
        hin_avail = is_tesseract_language_available("hin")
    except Exception:
        tam_avail = False
        hin_avail = False
    if ext == "pdf":
        try:
            doc = fitz.open(file_path)
            sample = ""
            for i, page in enumerate(doc):
                sample += page.get_text("text") or ""
                if i >= 1: break
            lang = detect_language_simple(sample)
            if lang == "Tamil" and tam_avail: return "tam+eng"
            if lang == "Hindi" and hin_avail: return "hin+eng"
            if tam_avail: return "tam+eng"
            if hin_avail: return "hin+eng"
            return "eng"
        except Exception:
            return "tam+eng" if tam_avail else ("hin+eng" if hin_avail else "eng")
    if ext in ["png", "jpg", "jpeg"]:
        if tam_avail: return "tam+eng"
        if hin_avail: return "hin+eng"
        return "eng"
    return "eng"

def extract_text_from_pdf(file_path: str, ocr_lang: str = None, dpi: int = OCR_DPI_DEFAULT, ocr_engine: str = None) -> List[str]:
    doc = fitz.open(file_path)
    count = len(doc)
    texts: List[str] = [""] * count
    to_ocr = []
    for i in range(count):
        page = doc[i]
        text = page.get_text("text") or ""
        if text.strip():
            texts[i] = text
        else:
            to_ocr.append(i)

    def ocr_page(idx: int) -> str:
        p = doc[idx]
        pix = p.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        
        # Preprocess image for better OCR
        img = preprocess_image(img)
        
        if ocr_engine and ocr_engine.lower() == "easyocr" and easyocr:
            try:
                langs = []
                if ocr_lang:
                    if "tam" in ocr_lang: langs.append("ta")
                    if "hin" in ocr_lang: langs.append("hi")
                if not langs: langs = ["en"]
                reader = easyocr.Reader(langs, gpu=False)
                res = reader.readtext(img, detail=0)
                return "\n".join(res)
            except Exception: pass
        try:
            base = pytesseract.image_to_string(img, lang=ocr_lang) if ocr_lang else pytesseract.image_to_string(img)
            if (len(base.strip()) < 10) or ("tam" in (ocr_lang or "")) and not any(0x0B80 <= ord(c) <= 0x0BFF for c in base):
                best = base
                for deg in (90, 180, 270):
                    try:
                        rot = img.rotate(deg, expand=True)
                        txt = pytesseract.image_to_string(rot, lang=ocr_lang) if ocr_lang else pytesseract.image_to_string(rot)
                        if len(txt.strip()) > len(best.strip()): best = txt
                    except Exception: continue
                return best
            return base
        except Exception:
            try: return pytesseract.image_to_string(img)
            except Exception: return ""

    if to_ocr:
        workers = max(1, min(4, len(to_ocr)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(ocr_page, idx): idx for idx in to_ocr}
            for fut, idx in futures.items():
                try: texts[idx] = fut.result() or ""
                except Exception: texts[idx] = ""
    return texts

def extract_text_from_image(file_path: str, ocr_lang: str = None, ocr_engine: str = None) -> List[str]:
    image = Image.open(file_path)
    
    # Preprocess for better OCR
    image = preprocess_image(image)
    
    if ocr_engine and ocr_engine.lower() == "easyocr" and easyocr:
        try:
            langs = []
            if ocr_lang:
                if "tam" in ocr_lang: langs.append("ta")
                if "hin" in ocr_lang: langs.append("hi")
            if not langs: langs = ["en"]
            reader = easyocr.Reader(langs, gpu=False)
            res = reader.readtext(image, detail=0)
            return ["\n".join(res)]
        except Exception: pass
    try:
        base = pytesseract.image_to_string(image, lang=ocr_lang) if ocr_lang else pytesseract.image_to_string(image)
        if (len(base.strip()) < 10) or ("tam" in (ocr_lang or "")) and not any(0x0B80 <= ord(c) <= 0x0BFF for c in base):
            best = base
            for deg in (90, 180, 270):
                try:
                    rot = image.rotate(deg, expand=True)
                    txt = pytesseract.image_to_string(rot, lang=ocr_lang) if ocr_lang else pytesseract.image_to_string(rot)
                    if len(txt.strip()) > len(best.strip()): best = txt
                except Exception: continue
            return [best]
        return [base]
    except Exception:
        try: return [pytesseract.image_to_string(image)]
        except Exception: return [""]

def extract_text_from_word(file_path: str) -> List[str]:
    if not Document: return [""]
    doc = Document(file_path)
    return [para.text for para in doc.paragraphs if para.text.strip()]

def extract_text_from_excel(file_path: str) -> List[str]:
    if not pd: return [""]
    df = pd.read_excel(file_path)
    return [df.to_string()]

def extract_text_from_txt(file_path: str) -> List[str]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return [f.read()]
    except Exception:
        try:
            with open(file_path, "r", encoding="latin-1") as f:
                return [f.read()]
        except Exception:
            return [""]

def extract_text_by_filetype(file_path: str, ext: str, ocr_lang: str = None, ocr_engine: str = None) -> List[str]:
    if ext == "pdf":
        return extract_text_from_pdf(file_path, ocr_lang=ocr_lang, ocr_engine=ocr_engine)
    elif ext in ["png", "jpg", "jpeg"]:
        return extract_text_from_image(file_path, ocr_lang=ocr_lang, ocr_engine=ocr_engine)
    elif ext == "docx":
        return extract_text_from_word(file_path)
    elif ext in ["xls", "xlsx"]:
        return extract_text_from_excel(file_path)
    elif ext == "txt":
        return extract_text_from_txt(file_path)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

def extract_text(file_path: str, file_type: str = "auto", ocr_lang: str = "eng", ocr_engine: str = None) -> str:
    """
    Wrapper function to extract text from a file path, handling multiple file types.
    Returns combined text as a single string.
    """
    if file_type == "auto" or not file_type:
        file_type = detect_file_type(file_path)
    
    try:
        texts = extract_text_by_filetype(file_path, file_type, ocr_lang=ocr_lang, ocr_engine=ocr_engine)
        return "\n".join(texts)
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
        return ""
