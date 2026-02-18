
import sys
import os

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def test_auto_loading():
    model_name = "Helsinki-NLP/opus-mt-ta-en"
    print(f"Testing Auto loading for {model_name}...")
    
    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Loading model...")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        text = "வணக்கம்"
        print(f"Translating: {text}")
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model.generate(**inputs)
        translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Success! Result: {translated}")
    except Exception as e:
        print(f"❌ Failed: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_auto_loading()
