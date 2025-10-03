# src/preprocess.py
import re
import spacy

# Load spaCy English model
nlp = spacy.load('en_core_web_sm')

def clean_text(text):
    """
    Clean and preprocess text:
    - Remove leading/trailing spaces
    - Remove extra newlines and multiple spaces
    - Remove non-alphanumeric characters (keep basic punctuation)
    - Lemmatize words using spaCy
    """
    if not text:
        return ""

    # Step 1: Strip and normalize spaces
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)  # replace multiple spaces/newlines with single space
    text = re.sub(r'[^A-Za-z0-9,.!? ]+', '', text)  # remove unwanted symbols

    # Step 2: Lemmatize using spaCy
    doc = nlp(text)
    lemmatized_text = " ".join([token.lemma_ for token in doc])

    return lemmatized_text

# Optional: test code
if __name__ == "__main__":
    sample = """
    This is a sample legal document! It contains multiple spaces, newlines, and numbers like 1234.
    Running, walked, and flying should be lemmatized.
    """
    print("=== Original ===")
    print(sample)
    print("\n=== Cleaned ===")
    print(clean_text(sample))
