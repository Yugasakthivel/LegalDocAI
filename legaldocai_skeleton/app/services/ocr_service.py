import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import os

try:
    from app.config import settings
    if hasattr(settings, 'TESSERACT_CMD') and settings.TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD
except:
    pass

def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Preprocess image for better OCR accuracy.
    
    Args:
        image: PIL Image object
    
    Returns:
        Preprocessed PIL Image
    """
    image = image.convert('L')
    
    image = image.filter(ImageFilter.MedianFilter(size=3))
    
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    
    return image

def perform_ocr(image_path: str, lang: str = 'eng') -> str:
    """
    Perform OCR on image files using Tesseract.
    
    Features:
    - Image preprocessing for better accuracy
    - Support for multiple languages (default: English)
    - Error handling for missing Tesseract installation
    
    Args:
        image_path: Path to the image file
        lang: Tesseract language code (default: 'eng', use 'hin' for Hindi)
    
    Returns:
        OCR-extracted text
    """
    try:
        if not os.path.exists(image_path):
            return f"Error: Image file not found at {image_path}"
        
        image = Image.open(image_path)
        
        preprocessed_image = preprocess_image(image)
        
        text = pytesseract.image_to_string(preprocessed_image, lang=lang)
        
        return text.strip()
    
    except pytesseract.TesseractNotFoundError:
        return (
            "Error: Tesseract OCR not found. Please install Tesseract:\n"
            "- Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki\n"
            "- Linux: sudo apt-get install tesseract-ocr\n"
            "- Mac: brew install tesseract\n"
            "Set TESSERACT_CMD in .env to the tesseract executable path."
        )
    
    except Exception as e:
        return f"Error performing OCR: {str(e)}"
