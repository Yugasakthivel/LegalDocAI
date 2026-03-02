import os
import re
import shutil
import subprocess
try:
    import fitz
except Exception:
    fitz = None
try:
    import pdfplumber
except Exception:
    pdfplumber = None
try:
    import PyPDF2
except Exception:
    PyPDF2 = None
from PIL import Image, ImageEnhance, ImageFilter
try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None

import concurrent.futures
from typing import List, Optional
from fastapi import HTTPException
from backend.app.core.config import TESSERACT_CMD
_pt = None
def _get_pytesseract():
    global _pt
    if _pt is None:
        try:
            import pytesseract as _p
            try:
                if TESSERACT_CMD:
                    _p.pytesseract.tesseract_cmd = TESSERACT_CMD
                    try:
                        base_dir = os.path.dirname(TESSERACT_CMD)
                        tess_dir = os.path.join(base_dir, "tessdata")
                        if tess_dir and os.path.isdir(tess_dir):
                            os.environ.setdefault("TESSDATA_PREFIX", tess_dir)
                    except Exception:
                        pass
            except Exception:
                pass
            _pt = _p
        except Exception:
            _pt = None
    return _pt

_easyocr = None
_easyocr_readers = {} # Cache readers by language tuple
_easyocr_failed_lang_keys = set()  # Cache failed reader inits to avoid repeated errors
_easyocr_disabled = False

def _get_easyocr():
    global _easyocr, _easyocr_disabled
    if _easyocr_disabled:
        return None
    if _easyocr is None:
        try:
            import easyocr as _e
            _easyocr = _e
        except Exception:
            _easyocr = None
    return _easyocr

def _get_cached_reader(langs: List[str], gpu: bool = False):
    global _easyocr_readers, _easyocr_failed_lang_keys, _easyocr_disabled
    ec = _get_easyocr()
    if not ec:
        return None
    
    # Sort langs to ensure consistent key
    lang_key = tuple(sorted(langs))
    if lang_key in _easyocr_failed_lang_keys:
        return None
    if lang_key not in _easyocr_readers:
        try:
            # Only create one reader at a time to avoid memory issues
            _easyocr_readers[lang_key] = ec.Reader(list(lang_key), gpu=gpu)
        except Exception as e:
            _easyocr_failed_lang_keys.add(lang_key)
            err = str(e or "")
            # Permanent disable when model checkpoint is incompatible.
            if "size mismatch" in err.lower():
                _easyocr_disabled = True
            print(f"[DEBUG] EasyOCR disabled for {list(lang_key)}: {e}")
            return None
    return _easyocr_readers[lang_key]

_docx_Document = None
def _get_docx_document():
    global _docx_Document
    if _docx_Document is None:
        try:
            from docx import Document as _D
            _docx_Document = _D
        except Exception:
            _docx_Document = None
    return _docx_Document

_pandas = None
def _get_pandas():
    global _pandas
    if _pandas is None:
        try:
            import pandas as _pd
            _pandas = _pd
        except Exception:
            _pandas = None
    return _pandas

OCR_DPI_DEFAULT = 300

def deskew(image: np.ndarray) -> np.ndarray:
    """
    Deskew the image using OpenCV.
    """
    if cv2 is None:
        return image
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        coords = np.column_stack(np.where(gray > 0))
        angle = cv2.minAreaRect(coords)[-1]
        
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
            
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated
    except Exception:
        return image

def apply_adaptive_threshold(image: np.ndarray) -> np.ndarray:
    """
    Apply adaptive thresholding using OpenCV.
    """
    if cv2 is None:
        return image
    try:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return thresh
    except Exception:
        return image

def preprocess_image(img: Image.Image) -> Image.Image:
    """
    Preprocess image to improve OCR accuracy:
    1. Convert to grayscale
    2. Deskew (using OpenCV)
    3. Adaptive Thresholding (using OpenCV)
    4. Fallback to Pillow enhancements if OpenCV fails
    """
    try:
        if cv2 is not None and np is not None:
            # Convert PIL to OpenCV format
            open_cv_image = np.array(img.convert("RGB"))
            # Deskew
            open_cv_image = deskew(open_cv_image)
            # Grayscale & Adaptive Threshold
            open_cv_image = apply_adaptive_threshold(open_cv_image)
            # Convert back to PIL
            return Image.fromarray(open_cv_image)
        
        # Fallback to Pillow-only processing
        img = img.convert("L")
        img = ImageEnhance.Contrast(img).enhance(1.6)
        img = ImageEnhance.Sharpness(img).enhance(1.3)
        img = img.filter(ImageFilter.MedianFilter(size=3))
        return img
    except Exception:
        # Final fallback
        try:
            return img.convert("L")
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
            if 0x0C00 <= cp <= 0x0C7F:
                return "Telugu"
        return "English" if re.search(r"[A-Za-z]", text) else "Unknown"
    except Exception:
        return "Unknown"

def is_tesseract_language_available(lang: str) -> bool:
    try:
        pt = _get_pytesseract()
        if not pt:
            return False
        langs = pt.get_languages(config="")
        if langs:
            return lang in (langs or [])
        # Fallback: call tesseract directly to list languages
        try:
            exe = TESSERACT_CMD or "tesseract"
            out = subprocess.run([exe, "--list-langs"], capture_output=True, text=True)
            if out.returncode == 0:
                found = any(f"\n{lang}\n" in f"\n{line.strip()}\n" or line.strip().endswith(f"\\{lang}") for line in out.stdout.splitlines())
                return bool(found)
        except Exception:
            pass
    except Exception:
        try:
            pt = _get_pytesseract()
            if not pt:
                return False
            img = Image.new("RGB", (16, 16), (255, 255, 255))
            _ = pt.image_to_string(img, lang=lang)
            return True
        except Exception:
            return False

def resolve_ocr_lang_for_file(file_path: str, ext: str, requested: str = None) -> str:
    if requested and requested.lower() != "auto":
        return requested
    try:
        tam_avail = is_tesseract_language_available("tam")
        hin_avail = is_tesseract_language_available("hin")
        tel_avail = is_tesseract_language_available("tel")
    except Exception:
        tam_avail = False
        hin_avail = False
        tel_avail = False
    if ext == "pdf":
        try:
            if fitz:
                doc = fitz.open(file_path)
                sample = ""
                for i, page in enumerate(doc):
                    sample += page.get_text("text") or ""
                    if i >= 1:
                        break
                lang = detect_language_simple(sample)
                if lang == "Tamil" and tam_avail:
                    return "tam+eng"
                if lang == "Telugu" and tel_avail:
                    return "tel+eng"
                if lang == "Hindi" and hin_avail:
                    return "hin+eng"
                if tam_avail:
                    return "tam+eng"
                if tel_avail:
                    return "tel+eng"
                if hin_avail:
                    return "hin+eng"
        except Exception:
            pass
    if ext in ["png", "jpg", "jpeg"]:
        if tam_avail: return "tam+eng"
        if tel_avail: return "tel+eng"
        if hin_avail: return "hin+eng"
        return "eng"
    return "eng"

def extract_text_from_pdf(file_path: str, ocr_lang: str = None, dpi: int = OCR_DPI_DEFAULT, ocr_engine: str = None) -> List[str]:
    """
    Unified PDF ingestion:
    1. Try PyMuPDF (fitz) page-by-page.
    2. Detect if page is digital (has text) or scanned (no text).
    3. Trigger OCR only for scanned pages.
    4. Fallback to pdfplumber/PyPDF2 for structural text if fitz fails.
    """
    if not fitz:
        # Fallback to pdfplumber/PyPDF2 when PyMuPDF is unavailable
        combined = ""
        try:
            if pdfplumber:
                with pdfplumber.open(file_path) as pdf:
                    for p in pdf.pages:
                        t = p.extract_text() or ""
                        combined += (t + "\n")
        except Exception:
            pass
        if (not combined.strip()) and PyPDF2:
            try:
                with open(file_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for p in reader.pages:
                        t = p.extract_text() or ""
                        combined += (t + "\n")
            except Exception:
                pass
        return [combined] if combined.strip() else [""]

    try:
        doc = fitz.open(file_path)
    except Exception as e:
        # Final structural fallback if fitz cannot open the file
        print(f"[DEBUG] fitz failed to open PDF: {e}")
        return _fallback_structural_extraction(file_path)

    count = len(doc)
    texts: List[str] = [""] * count
    to_ocr = []

    for i in range(count):
        try:
            page = doc[i]
            text = page.get_text("text") or ""
            # Threshold: If page has significant selectable text, treat as digital
            if len(text.strip()) > 20: 
                texts[i] = text
            else:
                to_ocr.append(i)
        except Exception:
            to_ocr.append(i)

    def ocr_page(idx: int) -> str:
        try:
            p = doc[idx]
            pix = p.get_pixmap(dpi=dpi)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            img = preprocess_image(img)
            
            # Engine selection
            if ocr_engine and ocr_engine.lower() == "easyocr":
                return _run_easyocr(img, ocr_lang)
            
            # Primary: Tesseract
            return _run_tesseract(img, ocr_lang)
        except Exception as e:
            print(f"[DEBUG] OCR failed for page {idx}: {e}")
            return ""

    if to_ocr:
        workers = max(1, min(4, len(to_ocr)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(ocr_page, idx): idx for idx in to_ocr}
            for fut, idx in futures.items():
                res = fut.result()
                if res and res.strip():
                    texts[idx] = res

    # If PyMuPDF + OCR returned nothing, try one last structural sweep
    if not any(t.strip() for t in texts):
        return _fallback_structural_extraction(file_path)

    return texts

def _fallback_structural_extraction(file_path: str) -> List[str]:
    combined_struct = ""
    try:
        if pdfplumber:
            with pdfplumber.open(file_path) as pdf:
                for p in pdf.pages:
                    t = p.extract_text() or ""
                    combined_struct += (t + "\n")
    except Exception:
        pass
    if (not combined_struct.strip()) and PyPDF2:
        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for p in reader.pages:
                    t = p.extract_text() or ""
                    combined_struct += (t + "\n")
        except Exception:
            pass
    return [combined_struct] if combined_struct.strip() else [""]

def _run_tesseract(img: Image.Image, ocr_lang: str) -> str:
    pt = _get_pytesseract()
    if not pt: return ""
    cfg = "--psm 3 --oem 1 -c preserve_interword_spaces=1"
    base = pt.image_to_string(img, lang=ocr_lang, config=cfg) if ocr_lang else pt.image_to_string(img, config=cfg)
    
    # Heuristic: If Tesseract fails to find Tamil in a Tamil doc, try rotation
    if ("tam" in (ocr_lang or "")) and not any(0x0B80 <= ord(c) <= 0x0BFF for c in base):
        best = base
        for deg in (90, 180, 270):
            try:
                rot = img.rotate(deg, expand=True)
                txt = pt.image_to_string(rot, lang=ocr_lang, config=cfg)
                if len(txt.strip()) > len(best.strip()): best = txt
            except Exception: continue
        return best
    return base

def _run_easyocr(img: Image.Image, ocr_lang: str) -> str:
    try:
        langs = []
        if ocr_lang:
            if "tam" in ocr_lang: langs.append("ta")
            if "hin" in ocr_lang: langs.append("hi")
            if "tel" in ocr_lang: langs.append("te")
        if not langs: langs = ["en"]
        reader = _get_cached_reader(langs, gpu=False)
        if reader:
            res = reader.readtext(img, detail=0)
            return "\n".join(res)
    except Exception: pass
    return ""

def extract_text_from_image(file_path: str, ocr_lang: str = None, ocr_engine: str = None) -> List[str]:
    image = Image.open(file_path)
    
    w, h = image.size
    if max(w, h) < 1200:
        scale = min(2.0, 1200.0 / float(max(w, h)))
        new_size = (int(w * scale), int(h * scale))
        image = image.resize(new_size, Image.LANCZOS)
    
    image = preprocess_image(image)
    
    try:
        if ocr_lang and (("tam" in ocr_lang) or ("hin" in ocr_lang) or ("tel" in ocr_lang)):
            ec = _get_easyocr()
            if ec:
                langs = []
                if "tam" in ocr_lang: langs.append("ta")
                if "hin" in ocr_lang: langs.append("hi")
                if "tel" in ocr_lang: langs.append("te")
                if not langs: langs = ["en"]
                reader = _get_cached_reader(langs, gpu=False)
                if reader:
                    res = reader.readtext(image, detail=0)
                    joined = "\n".join(res)
                    if joined.strip():
                        return [joined]
    except Exception:
        pass
    try:
        exe = TESSERACT_CMD or "tesseract"
        base = os.path.dirname(TESSERACT_CMD) if TESSERACT_CMD else ""
        td = os.path.join(base, "tessdata") if base else ""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            image.save(tmp.name, format="PNG")
            args = [exe, tmp.name, "stdout"]
            if ocr_lang: args += ["-l", ocr_lang]
            args += ["--psm", "3", "--oem", "1"]
            if td and os.path.isdir(td): args += ["--tessdata-dir", td]
            out = subprocess.run(args, capture_output=True, text=True)
            if out.returncode == 0 and (out.stdout or "").strip():
                return [out.stdout]
    except Exception:
        pass

    if ocr_engine and ocr_engine.lower() == "easyocr":
        try:
            langs = []
            if ocr_lang:
                if "tam" in ocr_lang: langs.append("ta")
                if "hin" in ocr_lang: langs.append("hi")
                if "tel" in ocr_lang: langs.append("te")
            if not langs: langs = ["en"]
            
            reader = _get_cached_reader(langs, gpu=False)
            if reader:
                res = reader.readtext(image, detail=0)
                return ["\n".join(res)]
        except Exception: pass
    try:
        pt = _get_pytesseract()
        if not pt:
            raise RuntimeError("pytesseract not available")
        tess_dir = None
        try:
            if TESSERACT_CMD:
                base = os.path.dirname(TESSERACT_CMD)
                td = os.path.join(base, "tessdata")
                if os.path.isdir(td):
                    tess_dir = td
        except Exception:
            pass
        cfg = "--psm 3 --oem 1 -c preserve_interword_spaces=1" + (f" --tessdata-dir {tess_dir}" if tess_dir else "")
        base = pt.image_to_string(image, lang=ocr_lang, config=cfg) if ocr_lang else pt.image_to_string(image, config=cfg)
        if (len(base.strip()) < 10) or ("tam" in (ocr_lang or "")) and not any(0x0B80 <= ord(c) <= 0x0BFF for c in base):
            best = base
            for deg in (90, 180, 270):
                try:
                    rot = image.rotate(deg, expand=True)
                    txt = pt.image_to_string(rot, lang=ocr_lang, config=cfg) if ocr_lang else pt.image_to_string(rot, config=cfg)
                    if len(txt.strip()) > len(best.strip()): best = txt
                except Exception: continue
            if len(best.strip()) < 10:
                for th in (110, 130, 150):
                    try:
                        gray = image.convert("L")
                        thr = gray.point(lambda x: 0 if x < th else 255, "1")
                        txt2 = pt.image_to_string(thr, lang=ocr_lang, config=cfg) if ocr_lang else pt.image_to_string(thr, config=cfg)
                        if len(txt2.strip()) > len(best.strip()): best = txt2
                    except Exception:
                        continue
            if len(best.strip()) < 10:
                ec = _get_easyocr()
                if ec:
                    langs = []
                    if ocr_lang:
                        if "tam" in ocr_lang: langs.append("ta")
                        if "hin" in ocr_lang: langs.append("hi")
                        if "tel" in ocr_lang: langs.append("te")
                    if not langs: langs = ["en"]
                    reader = _get_cached_reader(langs, gpu=False)
                    if reader:
                        res = reader.readtext(image, detail=0)
                        return ["\n".join(res)]
            return [best]
        return [base]
    except Exception:
        try:
            pt = _get_pytesseract()
            if pt:
                return [pt.image_to_string(image)]
            ec = _get_easyocr()
            if ec:
                langs = []
                if ocr_lang:
                    if "tam" in ocr_lang: langs.append("ta")
                    if "hin" in ocr_lang: langs.append("hi")
                    if "tel" in ocr_lang: langs.append("te")
                if not langs: langs = ["en"]
                reader = ec.Reader(langs, gpu=False)
                res = reader.readtext(image, detail=0)
                return ["\n".join(res)]
            return [""]
        except Exception:
            return [""]

def extract_text_from_word(file_path: str) -> List[str]:
    D = _get_docx_document()
    if not D: return [""]
    doc = D(file_path)
    return [para.text for para in doc.paragraphs if para.text.strip()]

def extract_text_from_excel(file_path: str) -> List[str]:
    pd = _get_pandas()
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

def extract_text(file_path: str, file_type: str = "auto", ocr_lang: str = "eng", ocr_engine: str = None) -> dict:
    """
    Wrapper function to extract text from a file path, handling multiple file types.
    Returns a dictionary with text and metadata (e.g., num_pages).
    """
    if file_type == "auto" or not file_type:
        file_type = detect_file_type(file_path)
    
    try:
        texts = extract_text_by_filetype(file_path, file_type, ocr_lang=ocr_lang, ocr_engine=ocr_engine)
        return {
            "text": "\n".join(texts),
            "num_pages": len(texts)
        }
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
        return {"text": "", "num_pages": 0}
