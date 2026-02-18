
import sys
import os

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from backend.app.routes.ai import _translate_mymemory

def test_mymemory_translation():
    tamil_text = "வணக்கம், இது ஒரு சோதனை."  # "Hello, this is a test."
    print(f"Testing MyMemory translation for: {tamil_text}")
    
    try:
        translated = _translate_mymemory(tamil_text, "English", "tam")
        if translated:
            print("✅ MyMemory Translation Success!")
            print(f"Translated: {translated}")
        else:
            print("❌ MyMemory Translation returned None.")
    except Exception as e:
        print(f"❌ MyMemory Translation Failed: {e}")

if __name__ == "__main__":
    test_mymemory_translation()
