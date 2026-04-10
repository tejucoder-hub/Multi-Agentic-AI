from typing import List


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def chunk_documents(documents: List[str], chunk_size: int = 600, overlap: int = 100) -> List[str]:
    all_chunks: List[str] = []
    for doc in documents:
        all_chunks.extend(chunk_text(doc, chunk_size=chunk_size, overlap=overlap))
    return all_chunks