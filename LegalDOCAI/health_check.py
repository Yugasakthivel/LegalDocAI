"""
Health Check Utility for LegalDocAI Backend
This script validates all services and dependencies
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.11+"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python version: {version.major}.{version.minor}.{version.micro} (Need 3.11+)")
        return False

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        "fastapi",
        "uvicorn",
        "pymongo",
        "spacy",
        "transformers",
        "torch",
        "pytesseract",
        "PIL",
        "numpy",
        "sklearn"
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ Package installed: {package}")
        except ImportError:
            print(f"✗ Package missing: {package}")
            missing.append(package)
    
    return len(missing) == 0

def check_env_file():
    """Check if .env file exists and has required variables"""
    if not Path(".env").exists():
        print("✗ .env file not found")
        return False
    
    print("✓ .env file exists")
    
    # Load and check critical variables
    from dotenv import load_dotenv
    load_dotenv()
    
    critical_vars = ["MONGO_URI", "DB_NAME"]
    optional_vars = ["OPENAI_API_KEY", "TESSERACT_CMD"]
    
    all_set = True
    for var in critical_vars:
        value = os.getenv(var)
        if value:
            print(f"  ✓ {var} is set")
        else:
            print(f"  ✗ {var} is missing")
            all_set = False
    
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            print(f"  ✓ {var} is set")
        else:
            print(f"  ⚠ {var} is not set (optional feature)")
    
    return all_set

def check_directories():
    """Check if required directories exist"""
    required_dirs = ["uploads", "uploads/processed", "chroma", "backend/app"]
    
    all_exist = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✓ Directory exists: {dir_path}")
        else:
            print(f"✗ Directory missing: {dir_path}")
            all_exist = False
    
    return all_exist

def check_spacy_model():
    """Check if spaCy English model is downloaded"""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("✓ spaCy model 'en_core_web_sm' is installed")
        return True
    except OSError:
        print("✗ spaCy model 'en_core_web_sm' not found")
        print("  Run: python -m spacy download en_core_web_sm")
        return False
    except Exception as e:
        print(f"✗ Error loading spaCy model: {e}")
        return False

def check_mongodb():
    """Check MongoDB connection"""
    try:
        from pymongo import MongoClient
        from backend.app.core.config import MONGO_URI
        
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        client.server_info()  # Trigger connection
        print(f"✓ MongoDB is reachable at {MONGO_URI}")
        client.close()
        return True
    except Exception as e:
        print(f"⚠ MongoDB not reachable: {e}")
        print("  App will use JSON file fallback")
        return False

def check_tesseract():
    """Check if Tesseract OCR is available"""
    try:
        import pytesseract
        from PIL import Image
        from backend.app.core.config import TESSERACT_CMD
        
        if TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        
        # Try to get version
        version = pytesseract.get_tesseract_version()
        print(f"✓ Tesseract OCR is available: v{version}")
        return True
    except Exception as e:
        print(f"⚠ Tesseract OCR not available: {e}")
        print("  OCR on images may not work")
        return False

def check_openai():
    """Check if OpenAI is configured"""
    try:
        from backend.app.core.config import OPENAI_API_KEY
        
        if OPENAI_API_KEY and OPENAI_API_KEY.startswith("sk-"):
            print("✓ OpenAI API key is configured")
            return True
        else:
            print("⚠ OpenAI API key not configured")
            print("  RAG/chat features will not work")
            return False
    except Exception as e:
        print(f"⚠ Error checking OpenAI config: {e}")
        return False

def main():
    """Run all health checks"""
    print("=" * 50)
    print("LegalDocAI Backend - Health Check")
    print("=" * 50)
    print()
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Environment File", check_env_file),
        ("Directories", check_directories),
        ("spaCy Model", check_spacy_model),
        ("MongoDB", check_mongodb),
        ("Tesseract OCR", check_tesseract),
        ("OpenAI Configuration", check_openai),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n[{name}]")
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ Error during check: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    
    critical_failed = []
    warnings = []
    
    for name, result in results:
        if name in ["Python Version", "Dependencies", "Environment File", "Directories"]:
            if not result:
                critical_failed.append(name)
        else:
            if not result:
                warnings.append(name)
    
    if critical_failed:
        print(f"\n✗ Critical checks failed: {len(critical_failed)}")
        for name in critical_failed:
            print(f"  - {name}")
        print("\nPlease fix these issues before running the application.")
        sys.exit(1)
    elif warnings:
        print(f"\n⚠ Some optional features unavailable: {len(warnings)}")
        for name in warnings:
            print(f"  - {name}")
        print("\nApplication can run with limited functionality.")
        sys.exit(0)
    else:
        print("\n✅ All checks passed! Backend is ready to run.")
        sys.exit(0)

if __name__ == "__main__":
    main()
