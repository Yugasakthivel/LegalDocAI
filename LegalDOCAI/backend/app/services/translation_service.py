import os
import torch
import asyncio
from typing import Optional, List, Dict
from transformers import MarianMTModel, MarianTokenizer

class TranslationService:
    _instance = None
    _models: Dict[str, MarianMTModel] = {}
    _tokenizers: Dict[str, MarianTokenizer] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TranslationService, cls).__new__(cls)
        return cls._instance

    def _get_model_name(self, source_lang: str, target_lang: str = "en") -> Optional[str]:
        source = source_lang.lower()
        target = target_lang.lower()
        
        if target in ["en", "english"]:
            if "tam" in source or "tamil" in source:
                return "Helsinki-NLP/opus-mt-ta-en"
            if "hin" in source or "hindi" in source:
                return "Helsinki-NLP/opus-mt-hi-en"
            if "tel" in source or "telugu" in source:
                return "Helsinki-NLP/opus-mt-te-en"
            if "kan" in source or "kannada" in source:
                return "Helsinki-NLP/opus-mt-kn-en"
            if "mal" in source or "malayalam" in source:
                return "Helsinki-NLP/opus-mt-ml-en"
        return None

    def load_model(self, model_name: str):
        if model_name not in self._models:
            print(f"[TranslationService] Loading model: {model_name}")
            try:
                self._tokenizers[model_name] = MarianTokenizer.from_pretrained(model_name)
                self._models[model_name] = MarianMTModel.from_pretrained(model_name)
                # Move to GPU if available
                if torch.cuda.is_available():
                    self._models[model_name] = self._models[model_name].to("cuda")
            except Exception as e:
                print(f"[TranslationService] Error loading {model_name}: {e}")
                return False
        return True

    async def translate_local(self, text: str, source_lang: str, target_lang: str = "en") -> str:
        model_name = self._get_model_name(source_lang, target_lang)
        if not model_name:
            return text

        if not self.load_model(model_name):
            return text

        tokenizer = self._tokenizers[model_name]
        model = self._models[model_name]
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Sentence-safe chunking should be handled before calling this, 
        # but we'll do basic character-based chunking here if needed.
        # For production, we'll use the sentence-safe chunker from nlp.py.
        
        try:
            # Tokenize and generate
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
            
            with torch.no_grad():
                translated_tokens = model.generate(
                    **inputs, 
                    max_length=512,
                    num_beams=4,
                    repetition_penalty=2.5,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )
            
            result = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
            return result
        except Exception as e:
            print(f"[TranslationService] Translation error: {e}")
            return text

    async def translate_batch(self, chunks: List[str], source_lang: str, target_lang: str = "en") -> List[str]:
        model_name = self._get_model_name(source_lang, target_lang)
        if not model_name or not self.load_model(model_name):
            return chunks

        tokenizer = self._tokenizers[model_name]
        model = self._models[model_name]
        device = "cuda" if torch.cuda.is_available() else "cpu"

        results = []
        # Process in batches of 4 for speed
        batch_size = 4
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            try:
                inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
                with torch.no_grad():
                    translated_tokens = model.generate(
                        **inputs,
                        max_length=512,
                        num_beams=4,
                        repetition_penalty=2.5,
                        no_repeat_ngram_size=3,
                        early_stopping=True
                    )
                decoded = [tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]
                results.extend(decoded)
            except Exception as e:
                print(f"[TranslationService] Batch translation error: {e}")
                results.extend(batch)
        
        return results

translation_service = TranslationService()
