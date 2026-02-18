
import asyncio
import os
import sys

# Add project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from backend.app.services.ocr_service import extract_text_from_image, resolve_ocr_lang_for_file
from backend.app.nlp import detect_language, translate_to_english

async def test_tamil_ocr():
    print("Starting Tamil OCR Test...")
    sample_path = os.path.join(project_root, "..", "legal_dataset", "raw_documents", "patta", "patta_002.png")
    sample_path = os.path.normpath(sample_path)
    
    if not os.path.exists(sample_path):
        print(f"Error: Sample file not found at {sample_path}")
        return

    print(f"File found: {sample_path}")
    
    # 1. Resolve OCR Language
    ocr_lang = resolve_ocr_lang_for_file(sample_path, "png")
    print(f"Resolved OCR Language: {ocr_lang}")
    
    # 2. Extract Text
    print("Extracting text...")
    text_list = extract_text_from_image(sample_path, ocr_lang=ocr_lang)
    text = "\n".join(text_list)
    print(f"Extracted Text Length: {len(text)}")
    print("--- Sample Extracted Text ---")
    print(text[:500])
    print("-----------------------------")
    
    # 3. Detect Language
    lang = detect_language(text)
    print(f"Detected Language: {lang}")
    
    # 4. Translate to English
    if lang == "Tamil":
        print("Translating to English...")
        translated = translate_to_english(text, src_lang="Tamil")
        print(f"Translated Text Length: {len(translated)}")
        print("--- Sample Translated Text ---")
        print(translated[:500])
        print("-----------------------------")
    else:
        print("Document not detected as Tamil. Skipping translation test.")

if __name__ == "__main__":
    asyncio.run(test_tamil_ocr())
