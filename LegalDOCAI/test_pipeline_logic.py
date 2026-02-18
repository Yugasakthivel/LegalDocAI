
import sys
import os
import asyncio
from unittest.mock import MagicMock

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from backend.app.routes.ai import _translate_mymemory, _translate_marian, _translate_libre

def test_pipeline_logic():
    src_text = "வணக்கம், இது ஒரு சோதனை." # Hello, this is a test.
    target_lang = "English"
    eff_lang = "tam"
    
    print(f"Testing pipeline logic for: {src_text}")
    
    translated = None
    warnings = []
    
    def try_local():
        nonlocal translated
        print("[DEBUG] Attempting local translation...")
        
        # Simulating the new logic in ai.py
        from backend.app.routes.ai import _source_code_from_hint
        source_code = _source_code_from_hint(eff_lang, src_text)
        if source_code in ["ta", "hi"]:
            print(f"[DEBUG] Priority: MyMemory for {source_code}")
            fb = _translate_mymemory(src_text, target_lang, eff_lang)
            if fb:
                print("[DEBUG] MyMemory success.")
                warnings.append("Used MyMemory for translation.")
                translated = fb
                return

        # Otherwise try MarianMT
        fb = _translate_marian(src_text, target_lang, eff_lang)
        if fb:
            print("[DEBUG] MarianMT success.")
            warnings.append("Used local MarianMT for translation.")
        else:
            print("[DEBUG] MarianMT failed, trying LibreTranslate...")
            fb = _translate_libre(src_text, target_lang)
            if fb: 
                warnings.append("Used LibreTranslate for translation.")
            else:
                print("[DEBUG] Local options failed, trying MyMemory...")
                fb = _translate_mymemory(src_text, target_lang, eff_lang)
                if fb: warnings.append("Used MyMemory for translation.")
        translated = fb

    try_local()
    
    print(f"Final Translated: {translated}")
    print(f"Warnings: {warnings}")
    
    if translated and "test" in translated.lower():
        print("✅ Pipeline Logic Test Passed!")
    else:
        print("❌ Pipeline Logic Test Failed!")

if __name__ == "__main__":
    test_pipeline_logic()
