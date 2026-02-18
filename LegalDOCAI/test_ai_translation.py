
import sys
import os

# Add the project root to sys.path to allow importing from backend
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.app.routes.ai import _translate_mymemory, _source_code_from_hint

def test_internal_translation():
    text = "உரிமையாளர்கள் பெயர்"
    source_hint = "tam+eng"
    target_lang = "English"
    
    print(f"Testing _source_code_from_hint with '{source_hint}':")
    src = _source_code_from_hint(source_hint, text)
    print(f"Result: {src}")
    
    print(f"\nTesting _translate_mymemory with text: '{text}'")
    translated = _translate_mymemory(text, target_lang, source_hint)
    print(f"Translated: {translated}")

if __name__ == "__main__":
    test_internal_translation()
