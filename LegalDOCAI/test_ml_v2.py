import sys
import os
sys.path.insert(0, os.getcwd())
try:
    from backend.app.services.ml_service import classifier
    print('is_loaded:', classifier.is_loaded)
    text = 'This is a sample sale deed with vendor and vendee and schedule property for consideration. ' * 5
    print('Testing text length:', len(text))
    pred, conf = classifier.predict_document_type(text)
    rule_cat = classifier.detect_document_type_by_rules(text)
    print(f'Prediction: {pred} (Conf: {conf})')
    print(f'Rule-based: {rule_cat}')
    
    # Simulate verify_document_ml
    res = classifier.verify_document_ml(text)
    print('Verification Result:', res)
except Exception as e:
    print('Error:', e)
