
import sys
import os

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from backend.app.routes.ai import _translate_mymemory

def test_mymemory_long_text():
    # Long Tamil text (repeated greeting)
    tamil_text = "வணக்கம், இது ஒரு சோதனை. " * 50 
    print(f"Testing MyMemory translation for long text ({len(tamil_text)} chars)")
    
    try:
        translated = _translate_mymemory(tamil_text, "English", "tam")
        if translated:
            print("✅ MyMemory Long Translation Success!")
            print(f"Translated length: {len(translated)}")
            print(f"Sample: {translated[:100]}...")
        else:
            print("❌ MyMemory Long Translation returned None.")
    except Exception as e:
        print(f"❌ MyMemory Long Translation Failed: {e}")

if __name__ == "__main__":
    test_mymemory_long_text()
