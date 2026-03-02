import os, sys
repo = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if repo not in sys.path:
    sys.path.append(repo)
from backend.app.core.config import UPLOAD_FOLDER, TESSERACT_CMD
from PIL import Image, ImageEnhance
import pytesseract
def test(doc_id, filename):
    p = os.path.join(UPLOAD_FOLDER, "documents", f"{doc_id}_{filename}")
    print("path:", p, "exists:", os.path.exists(p))
    if not os.path.exists(p):
        return
    try:
        base = os.path.dirname(TESSERACT_CMD or "")
        tess = os.path.join(base, "tessdata")
        if tess and os.path.isdir(tess):
            os.environ.setdefault("TESSDATA_PREFIX", tess)
    except Exception:
        pass
    img = Image.open(p)
    w,h = img.size
    if max(w,h) < 1200:
        s = min(2.0, 1200.0/float(max(w,h)))
        img = img.resize((int(w*s), int(h*s)), Image.LANCZOS)
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = ImageEnhance.Sharpness(img).enhance(1.5)
    out = ""
    tess = None
    try:
        base = os.path.dirname(TESSERACT_CMD or "")
        td = os.path.join(base, "tessdata")
        if os.path.isdir(td):
            tess = td
    except Exception:
        pass
    cfg = "--psm 6 --oem 1 -c preserve_interword_spaces=1" + (f" --tessdata-dir {tess}" if tess else "")
    try:
        out = pytesseract.image_to_string(img, lang="tam+eng", config=cfg)
        print("pytesseract chars:", len(out.strip()))
    except Exception as e:
        print("pytesseract error:", e)
    try:
        import easyocr
        reader = easyocr.Reader(["ta","en"], gpu=False)
        res = reader.readtext(img, detail=0)
        joined = "\n".join(res)
        print("easyocr chars:", len(joined.strip()))
    except Exception as e:
        print("easyocr error:", e)
    try:
        import subprocess
        temp_png = os.path.join(UPLOAD_FOLDER, "tmp_cli_ocr.png")
        img.save(temp_png, format="PNG")
        exe = TESSERACT_CMD or "tesseract"
        args = [exe, temp_png, "stdout", "-l", "tam+eng", "--psm", "6", "--oem", "1"]
        if tess: args += ["--tessdata-dir", tess]
        out = subprocess.run(args, capture_output=True, text=True)
        print("cli returncode:", out.returncode)
        print("cli chars:", len((out.stdout or "").strip()))
        if out.stderr: print("cli stderr:", out.stderr.strip()[:200])
        try:
            os.remove(temp_png)
        except Exception:
            pass
    except Exception as e:
        print("cli error:", e)
if __name__ == "__main__":
    doc_id = sys.argv[1]
    filename = sys.argv[2]
    test(doc_id, filename)
