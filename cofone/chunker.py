import re


def chunk_text(text, mode="smart", size=500, overlap=50):
    if mode == "smart":
        return _chunk_smart(text)
    elif mode == "sentences":
        return _chunk_sentences(text)
    elif mode == "paragraphs":
        return _chunk_paragraphs(text)
    return _chunk_fixed(text, size, overlap)


def _chunk_smart(text):
    paragraphs = _chunk_paragraphs(text)
    chunks = []
    for para in paragraphs:
        if len(para) > 600:
            chunks.extend(_chunk_sentences(para))
        else:
            chunks.append(para)
    return [c for c in chunks if c.strip()]


def _chunk_paragraphs(text):
    blocks = re.split(r"\n\s*\n", text)
    return [b.strip() for b in blocks if b.strip()]


def _chunk_sentences(text):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) < 500:
            current += " " + s
        else:
            if current:
                chunks.append(current.strip())
            current = s
    if current:
        chunks.append(current.strip())
    return chunks


def _chunk_fixed(text, size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks