from PIL import Image
import pytesseract
import pdfplumber


# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_from_image(image_path):
    """
    Extract text from an image file using Tesseract OCR.
    """
    try:
        text = pytesseract.image_to_string(Image.open("D:\Project\LegalDocAI\LegalDOCAI\data\document2.png"))
        return text
    except Exception as e:
        return f"Error extracting text from image: {e}"

def extract_text_from_pdf(pdf_path):
    """
    Extract text from all pages of a PDF file using pdfplumber.
    """
    text = ""
    try:
        with pdfplumber.open("D:\Project\LegalDocAI\LegalDOCAI\data\document1.pdf") as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        return f"Error extracting text from PDF: {e}"

# Example usage (uncomment to test directly)
# if __name__ == "__main__":
#     print("Image OCR:")
#     print(extract_text_from_image(r"D:\Project\document2.png"))
#     print("PDF OCR:")
#     print(extract_text_from_pdf(r"D:\Project\document1.pdf"))

