import spacy
try:
    nlp = spacy.load("en_core_web_sm")
    print("SUCCESS: en_core_web_sm loaded.")
    doc = nlp("This is a test.")
    print(f"Tokens: {[t.text for t in doc]}")
except Exception as e:
    print(f"FAILURE: {e}")
