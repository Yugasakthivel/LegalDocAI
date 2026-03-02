import re
from backend.app.services.analysis_service import transliterate_tamil_to_latin, extract_indian_entities

def is_english(s: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z\s]+", s or ""))

def test_transliterate_simple_tamil_name():
    # "விஜயராணி" approximate transliteration should be latin-only string
    src = "விஜயராணி"
    out = transliterate_tamil_to_latin(src)
    assert out and is_english(out)
    assert len(out.replace(" ", "")) >= 6

def test_extract_owner_names_from_patta_block():
    # Minimal patta-style snippet with Tamil owner header and name
    text = """
    உரிமையாளர் பெயர்
    விஜயராணி ச
    வருவாய் கிராமம் : அறியபாடி
    தாலுகா : திருவண்ணாமலை
    """
    entities = extract_indian_entities(text)
    parties = entities.get("parties") or []
    assert isinstance(parties, list)
    # Expect at least one latin-only name from transliteration
    assert any(is_english(p) and len(p.replace(" ", "")) >= 5 for p in parties)
