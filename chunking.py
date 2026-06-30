"""
Chunking utilities: splits parsed document text into overlapping chunks
suitable for embedding and retrieval.
"""

from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    source_file: str
    chunk_index: int
    text: str


def chunk_text(
    text: str,
    source_file: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    """
    Splits text into overlapping chunks based on character count.
    Tries to break on paragraph/sentence boundaries when possible
    so chunks don't cut words in half mid-sentence.
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    index = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        # Try to extend to the nearest paragraph break, then sentence break,
        # then whitespace, so we don't cut a word in half.
        if end < text_length:
            paragraph_break = text.rfind("\n\n", start, end)
            sentence_break = text.rfind(". ", start, end)
            space_break = text.rfind(" ", start, end)

            if paragraph_break != -1 and paragraph_break > start + chunk_size // 2:
                end = paragraph_break + 2
            elif sentence_break != -1 and sentence_break > start + chunk_size // 2:
                end = sentence_break + 2
            elif space_break != -1:
                end = space_break + 1

        chunk_str = text[start:end].strip()
        if chunk_str:
            chunks.append(
                Chunk(
                    chunk_id=f"{source_file}::chunk_{index}",
                    source_file=source_file,
                    chunk_index=index,
                    text=chunk_str,
                )
            )
            index += 1

        # Move start forward, accounting for overlap
        next_start = end - chunk_overlap
        start = next_start if next_start > start else end

    return chunks


def chunk_multiple_documents(
    parsed_docs: dict,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    """
    parsed_docs: dict mapping filename -> full parsed text.
    Returns a flat list of Chunk objects across all documents.
    """
    all_chunks = []
    for filename, text in parsed_docs.items():
        all_chunks.extend(
            chunk_text(text, filename, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        )
    return all_chunks
