import pytest
from backend.app.services.ml_service import DocumentClassifier
from unittest.mock import patch, MagicMock
import os

def test_model_loading_with_missing_files():
    with patch.object(DocumentClassifier, '_load_model', lambda self: None):
        classifier = DocumentClassifier()
    # Test when both model paths are missing
    with patch('backend.app.services.ml_service.os.path.exists', return_value=False):
        classifier._load_model()
        assert classifier.is_loaded == False
        assert classifier.model is None
        assert classifier.vectorizer is None

def test_model_loading_with_partial_files():
    with patch.object(DocumentClassifier, '_load_model', lambda self: None):
        classifier = DocumentClassifier()
    # Test when only one of model/vectorizer exists (ALT path absent)
    with patch('backend.app.services.ml_service.os.path.exists', side_effect=[False, True, False]):
        classifier._load_model()
        assert classifier.is_loaded == False
        assert classifier.model is None
        assert classifier.vectorizer is None

def test_model_loading_with_corrupted_files():
    with patch.object(DocumentClassifier, '_load_model', lambda self: None):
        classifier = DocumentClassifier()
    # Test when files exist but are corrupted
    with patch('backend.app.services.ml_service.os.path.exists', side_effect=[False, True, True]), \
         patch('backend.app.services.ml_service.joblib.load', side_effect=Exception("Corrupted file")):
        classifier._load_model()
        assert classifier.is_loaded == False
        assert classifier.model is None
        assert classifier.vectorizer is None

def test_document_type_classification_with_empty_text():
    classifier = DocumentClassifier()
    doc_type, confidence = classifier.predict_document_type("")
    assert doc_type == "Generic Legal Document"
    assert confidence == 0.0

def test_document_type_classification_with_short_text():
    classifier = DocumentClassifier()
    doc_type, confidence = classifier.predict_document_type("short")
    assert doc_type == "Unknown"
    assert confidence == 0.0

def test_document_type_classification_with_none_text():
    classifier = DocumentClassifier()
    doc_type, confidence = classifier.predict_document_type(None)
    assert doc_type == "Generic Legal Document"
    assert confidence == 0.0

def test_rule_based_classification_with_empty_text():
    classifier = DocumentClassifier()
    doc_type = classifier.detect_document_type_by_rules("")
    assert doc_type is None

def test_rule_based_classification_with_none_text():
    classifier = DocumentClassifier()
    doc_type = classifier.detect_document_type_by_rules(None)
    assert doc_type is None

def test_verify_document_with_empty_text():
    classifier = DocumentClassifier()
    result = classifier.verify_document_ml("")
    assert result["marker"] == "Rejected"
    assert result["ai_confidence"] == 0
    assert any("Model could not classify document" in issue for issue in result["issues"])

def test_verify_document_with_none_text():
    classifier = DocumentClassifier()
    result = classifier.verify_document_ml(None)
    assert result["marker"] == "Rejected"
    assert result["ai_confidence"] == 0
    assert any("Model could not classify document" in issue for issue in result["issues"])

def test_train_model_with_empty_data():
    classifier = DocumentClassifier()
    result = classifier.train_model([], [])
    assert result["success"] == False
    assert result["error"] == "No training data provided"

def test_train_model_with_none_data():
    classifier = DocumentClassifier()
    result = classifier.train_model(None, None)
    assert result["success"] == False
    assert result["error"] == "No training data provided"

def test_incremental_training_with_invalid_path():
    classifier = DocumentClassifier()
    result = classifier.train_model_incremental_from_path("/invalid/path", "csv")
    assert result["success"] == False
    assert "Dataset not found" in result["error"]

def test_incremental_training_with_empty_file():
    classifier = DocumentClassifier()
    with patch('backend.app.services.ml_service.os.path.exists', return_value=True):
        with patch('builtins.open', create=True) as mock_file:
            mock_file.return_value.__enter__.return_value = []
            result = classifier.train_model_incremental_from_path("/valid/path", "csv")
            assert result["success"] == False
            assert result["error"] == "No training data provided"

def test_get_supported_document_types():
    classifier = DocumentClassifier()
    doc_types = classifier.get_supported_document_types()
    assert len(doc_types) > 0
    assert "Patta" in doc_types
    assert "Aadhaar Card" in doc_types
    assert "Passport" in doc_types

def test_evaluate_dataset_with_empty_data():
    classifier = DocumentClassifier()
    result = classifier.evaluate_dataset([], [])
    assert result["success"] == False
    assert result["error"] == "No evaluation data provided"

def test_evaluate_dataset_with_none_data():
    classifier = DocumentClassifier()
    result = classifier.evaluate_dataset(None, None)
    assert result["success"] == False
    assert result["error"] == "No evaluation data provided"

def test_evaluate_dataset_with_unloaded_model():
    classifier = DocumentClassifier()
    classifier.is_loaded = False
    result = classifier.evaluate_dataset(["sample text"], ["label"])
    assert result["success"] == False
    assert result["error"] == "Model not loaded"
