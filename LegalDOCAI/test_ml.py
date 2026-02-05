import sys
import os
sys.path.insert(0, os.getcwd())
try:
    from backend.app.services.ml_service import classifier
    print('is_loaded:', classifier.is_loaded)
    print('predict test:', classifier.predict_document_type('This is a sample sale deed with parties and consideration'))
except Exception as e:
    print('Import or predict error:', e)
