import re
from typing import List

try:
    import tiktoken
except Exception:
    tiktoken = None


class BaseSplitter:
    def split(self, text: str) -> List[str]:
        raise NotImplementedError


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Za-zÁÉÍÓÚÜÑ0-9])", re.UNICODE)


class TokenAwareSentenceSplitter(BaseSplitter):
    """
    Splits text by sentences, keeping chunks below max_tokens.
    Uses tiktoken when available; falls back to character length otherwise.
    """
    def __init__(self, max_tokens: int = 900, overlap_tokens: int = 120, tokenizer_name: str = "cl100k_base"):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.encoder = None
        if tiktoken:
            try:
                self.encoder = tiktoken.get_encoding(tokenizer_name)
            except Exception:
                try:
                    self.encoder = tiktoken.encoding_for_model("gpt-4o-mini")
                except Exception:
                    self.encoder = None

    def _count(self, text: str) -> int:
        if not self.encoder:
            return len(text)
        return len(self.encoder.encode(text))

    def split(self, text: str) -> List[str]:
        if not text:
            return []
        sents = _SENT_SPLIT.split(text.strip())
        chunks, buf, cur = [], [], 0

        def flush():
            if buf:
                chunks.append(" ".join(buf).strip())

        for s in sents:
            n = self._count(s)
            if not buf:
                buf.append(s); cur = n; continue
            if cur + n <= self.max_tokens:
                buf.append(s); cur += n
            else:
                flush()
                if self.overlap_tokens > 0:
                    overlap, acc = [], 0
                    for prev in reversed(buf):
                        t = self._count(prev)
                        if acc + t <= self.overlap_tokens:
                            overlap.insert(0, prev)
                            acc += t
                        else:
                            break
                    buf = overlap
                    cur = sum(self._count(x) for x in buf)
                else:
                    buf, cur = [], 0
                buf.append(s); cur += n
        flush()
        return [c for c in chunks if c]