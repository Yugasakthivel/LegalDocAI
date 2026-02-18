
import sys
import os
import asyncio
from unittest.mock import MagicMock, patch

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

# Mock dependencies that might fail in this environment
sys.modules['transformers'] = MagicMock()
sys.modules['torch'] = MagicMock()

import httpx
from backend.app.routes.ai import translate_analyze
from fastapi import UploadFile
import io

async def test_translate_analyze_full():
    # Simulate a file upload
    content = "வணக்கம், இது ஒரு சோதனை.".encode("utf-8")
    file = UploadFile(filename="test.txt", file=io.BytesIO(content))
    
    print("Testing translate_analyze full pipeline...")
    
    # We need to mock get_current_user and other dependencies if they are called
    # But translate_analyze takes them as arguments, we can pass None or Mocks
    
    try:
        # Mocking the background task and user
        result = await translate_analyze(
            file=file,
            target_lang="English",
            ocr_lang="tam",
            engine="local",
            user={"email": "test@example.com"}
        )
        
        print(f"Result Keys: {result.keys()}")
        print(f"Translated Text: {result.get('translated_text')}")
        print(f"Warnings: {result.get('analytics', {}).get('warnings', [])}")
        
        if result.get('translated_text') and "test" in result.get('translated_text').lower():
            print("✅ Full Pipeline Test Passed!")
        else:
            print("❌ Full Pipeline Test Failed!")
            
    except Exception as e:
        print(f"❌ Test Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_translate_analyze_full())
