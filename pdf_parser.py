"""
PDF parsing using LlamaParse (LlamaIndex's cloud-based document parser).

Requires a LlamaCloud API key set as the LLAMA_CLOUD_API_KEY environment
variable. Get one for free at https://cloud.llamaindex.ai
"""

import os
import tempfile

from dotenv import load_dotenv

load_dotenv()

try:
    from llama_parse import LlamaParse
except Exception:  # pragma: no cover - handled gracefully at runtime
    LlamaParse = None


def _get_api_key() -> str:
    for name in ("LLAMA_CLOUD_API_KEY", "LLAMA_PARSE_KEY", "llama_parse_key"):
        value = os.getenv(name)
        if value:
            return value
    raise RuntimeError(
        "No LlamaParse API key found. Set LLAMA_CLOUD_API_KEY, LLAMA_PARSE_KEY, or llama_parse_key in your .env file."
    )


def get_parser():
    if LlamaParse is None:
        raise RuntimeError("The llama-parse package is not available. Install it with pip install llama-parse.")

    api_key = _get_api_key()
    return LlamaParse(
        api_key=api_key,
        result_type="markdown",  # "markdown" preserves structure better than "text"
        verbose=False,
    )


def parse_pdf_bytes(file_bytes: bytes, filename: str) -> str:
    """
    Parses a single PDF (given as raw bytes) using LlamaParse and
    returns the extracted text/markdown content as a single string.
    """
    parser = get_parser()

    # LlamaParse needs a real file path, so write to a temp file first.
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        documents = parser.load_data(tmp_path)
        full_text = "\n\n".join(doc.text for doc in documents if doc.text)
        return full_text
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def parse_multiple_pdfs(uploaded_files) -> dict:
    """
    Takes a list of Streamlit UploadedFile objects and returns a dict
    mapping filename -> parsed text content.
    """
    results = {}
    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        text = parse_pdf_bytes(file_bytes, uploaded_file.name)
        results[uploaded_file.name] = text
    return results
