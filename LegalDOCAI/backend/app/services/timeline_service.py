import re
import dateparser
import spacy
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from backend.app.nlp import _get_nlp

class TimelineService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TimelineService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Common date patterns in legal docs (e.g., 12th Jan 2023, 01/01/2024, etc.)
        self.date_pattern = r"(?i)\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{2,4}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{2,4})\b"
        
        # Relative date keywords
        self.relative_keywords = ["today", "yesterday", "tomorrow", "next week", "last month", "current year"]

    def extract_timeline(self, text: str) -> Dict[str, Any]:
        """
        Main pipeline for Module 8: Timeline Extraction.
        """
        if not text or not text.strip():
            return {"timeline": [], "metadata": {"status": "empty"}}

        # 1. Sentence Segmentation
        nlp = _get_nlp()
        if nlp:
            doc = nlp(text[:20000]) # Process in reasonable chunks
            sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
        else:
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 10]

        # 2 & 3. Temporal ER & Normalization
        raw_events = []
        for i, sent in enumerate(sentences):
            # Check for explicit dates
            dates = re.findall(self.date_pattern, sent)
            
            # Context window for cross-sentence linking
            # If no date in current sentence, check if previous sentence had a date
            # (Often seen in legal lists: "On 12/01/2023: \n - Item A \n - Item B")
            if not dates and i > 0:
                prev_dates = re.findall(self.date_pattern, sentences[i-1])
                if prev_dates:
                    # Only link if the current sentence looks like a continuation or list item
                    if sent.strip().startswith(("-", "*", "•")) or len(sent.split()) < 10:
                        dates = [prev_dates[-1]]

            for date_str in dates:
                normalized_date = self.normalize_date(date_str)
                if normalized_date:
                    event_desc = self.clean_event_description(sent, date_str)
                    if event_desc:
                        raw_events.append({
                            "date": normalized_date,
                            "original_date": date_str,
                            "event": event_desc,
                            "confidence": 0.85 if date_str in sent else 0.65 # Lower confidence if linked from prev sentence
                        })

        # 4. Deduplication & Sorting
        # Group by normalized date and unique events
        unique_events = {}
        for ev in raw_events:
            key = (ev["date"], ev["event"].lower())
            if key not in unique_events or ev["confidence"] > unique_events[key]["confidence"]:
                unique_events[key] = ev

        sorted_timeline = sorted(unique_events.values(), key=lambda x: x["date"])

        # 5. Output Structure
        return {
            "timeline": sorted_timeline,
            "metadata": {
                "event_count": len(sorted_timeline),
                "start_date": sorted_timeline[0]["date"] if sorted_timeline else None,
                "end_date": sorted_timeline[-1]["date"] if sorted_timeline else None,
                "engine": "TimelineService-v1.0"
            }
        }

    def normalize_date(self, date_str: str) -> Optional[str]:
        """
        Converts diverse date formats into ISO YYYY-MM-DD.
        """
        try:
            # Handle Indian/British formats (DD/MM/YYYY) which dateparser sometimes misinterprets
            # If it matches DD/MM/YYYY, try to force that order
            if re.match(r"^\d{1,2}/\d{1,2}/\d{2,4}$", date_str):
                parsed = dateparser.parse(date_str, settings={'DATE_ORDER': 'DMY', 'PREFER_DAY_OF_MONTH': 'first'})
            else:
                parsed = dateparser.parse(date_str, settings={'PREFER_DAY_OF_MONTH': 'first'})
            
            if parsed:
                return parsed.strftime("%Y-%m-%d")
        except Exception:
            pass
        return None

    def clean_event_description(self, sentence: str, date_str: str) -> str:
        """
        Extracts a meaningful event description from a sentence by removing the date string
        and cleaning up punctuation/extra spaces.
        """
        # Remove the date string from the sentence
        desc = sentence.replace(date_str, "").strip()
        
        # Clean up leading/trailing punctuation often left behind
        desc = re.sub(r"^[,\s:;.\-]+", "", desc)
        desc = re.sub(r"[,\s:;.\-]+$", "", desc)
        
        # If the remaining description is too short, use the whole sentence
        if len(desc.split()) < 3:
            return sentence.strip()
            
        return desc

timeline_service = TimelineService()
