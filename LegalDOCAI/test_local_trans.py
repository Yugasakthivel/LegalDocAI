
import sys
import os

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from backend.app.routes.ai import _translate_marian, _marian_model_for

def test_local_translation():
    tamil_text = "வணக்கம், இது ஒரு சோதனை."  # "Hello, this is a test."
    target = "English"
    hint = "tam"
    
    print(f"Testing local MarianMT translation for: {tamil_text}")
    model_name = _marian_model_for(target, hint)
    print(f"Model name selected: {model_name}")
    
    if not model_name:
        print("❌ Model name selection failed.")
        return

    try:
        from transformers import MarianMTModel, MarianTokenizer
        import torch
        print(f"Loading tokenizer for {model_name}...")
        tok = MarianTokenizer.from_pretrained(model_name)
        print(f"Loading model for {model_name}...")
        mdl = MarianMTModel.from_pretrained(model_name)
        
        print("Running translation...")
        enc = tok([tamil_text], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            gen = mdl.generate(**enc, max_length=512)
        translated = tok.decode(gen[0], skip_special_tokens=True)
        
        print("✅ Local Translation Success!")
        print(f"Translated: {translated}")
    except Exception as e:
        print(f"❌ Local Translation Failed with error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_local_translation()
