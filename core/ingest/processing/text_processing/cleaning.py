import re

_HYPHEN_LINEBREAK = re.compile(r"(\w)-\s*\n\s*(\w)")
_MULTI_SPACES = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINES = re.compile(r"\n{3,}")
_WS_AROUND_NEWLINE = re.compile(r"[ \t]*\n[ \t]*")

def normalize_text(text: str) -> str:
    """Cleans common OCR / linebreak issues."""
    if not text:
        return ""
    text = _HYPHEN_LINEBREAK.sub(r"\1\2", text)
    text = _WS_AROUND_NEWLINE.sub("\n", text)
    text = _MULTI_SPACES.sub(" ", text)
    text = _MULTI_NEWLINES.sub("\n\n", text)
    return text.strip()