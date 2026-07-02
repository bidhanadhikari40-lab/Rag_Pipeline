"""
Streamlit RAG Playground
=========================
Upload JSON, Markdown, or PDF -> parse (PDF only) -> chunk -> watch the chunking
process -> ask questions -> see retrieved chunks -> get an answer.

Run with:
    streamlit run app.py
"""

import io
import json
import logging
import time
from pathlib import Path

import numpy as np
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Optional dependencies (only needed for specific features). Imported lazily
# with friendly errors so the app still loads if one is missing.
# ---------------------------------------------------------------------------
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    import anthropic
except ImportError:
    anthropic = None


# ===========================================================================
# Page config
# ===========================================================================
st.set_page_config(page_title="RAG Playground", page_icon="🔎", layout="wide")

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("rag_playground")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

logger.info("Starting RAG Playground app")

if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""
if "source_type" not in st.session_state:
    st.session_state.source_type = None
if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None
if "tfidf_matrix" not in st.session_state:
    st.session_state.tfidf_matrix = None
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []


# ===========================================================================
# Ingestion helpers
# ===========================================================================
def extract_text_from_pdf(file_bytes, progress_container):
    """Parse a PDF page by page, showing progress as it goes."""
    if PdfReader is None:
        st.error("pypdf is not installed. Run: pip install pypdf")
        return ""

    reader = PdfReader(io.BytesIO(file_bytes))
    n_pages = len(reader.pages)
    logger.info("Starting PDF parse: %s pages", n_pages)
    bar = progress_container.progress(0, text="Starting PDF parse...")
    pages_text = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages_text.append(text)
        pct = int(((i + 1) / n_pages) * 100)
        bar.progress(pct, text=f"Parsing page {i + 1}/{n_pages}...")
        time.sleep(0.02)  # tiny delay so the progress bar is visibly animated

    bar.progress(100, text=f"Parsed {n_pages} page(s).")
    return "\n\n".join(pages_text)


def json_to_text(file_bytes):
    """Turn uploaded JSON into a readable text blob (pretty-printed)."""
    data = json.loads(file_bytes.decode("utf-8"))
    return json.dumps(data, indent=2, ensure_ascii=False)


def md_to_text(file_bytes):
    return file_bytes.decode("utf-8")


# ===========================================================================
# Chunking
# ===========================================================================
def chunk_fixed_size(text, chunk_size, overlap):
    chunks = []
    step = max(chunk_size - overlap, 1)
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append({"start": start, "end": end, "text": text[start:end]})
        if end == n:
            break
        start += step
    return chunks


def chunk_by_paragraph(text, max_chars):
    """Split on blank lines / paragraph breaks, merging small paragraphs up
    to max_chars so chunks aren't too tiny or too huge."""
    raw_paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    buf = ""
    cursor = 0
    for para in raw_paras:
        idx = text.find(para, cursor)
        if idx == -1:
            idx = cursor
        if len(buf) + len(para) + 1 <= max_chars:
            buf = (buf + "\n\n" + para).strip() if buf else para
        else:
            if buf:
                start = text.find(buf[:30], 0) if buf else idx
                chunks.append({"start": start, "end": start + len(buf), "text": buf})
            buf = para
        cursor = idx + len(para)
    if buf:
        start = text.find(buf[:30], 0)
        chunks.append({"start": start, "end": start + len(buf), "text": buf})
    return chunks


def run_chunking_with_visualization(text, strategy, chunk_size, overlap, viz_container):
    """Compute chunks, then replay their creation visually so the user can
    watch the chunking process happen."""
    logger.info("Chunking text of length %s with strategy %s", len(text), strategy)
    if strategy == "Fixed-size (characters)":
        raw_chunks = chunk_fixed_size(text, chunk_size, overlap)
    else:
        raw_chunks = chunk_by_paragraph(text, chunk_size)

    bar = viz_container.progress(0, text="Starting chunking...")
    log = viz_container.empty()
    lines = []
    total = len(raw_chunks)

    for i, c in enumerate(raw_chunks):
        preview = c["text"][:80].replace("\n", " ")
        lines.append(f"**Chunk {i + 1}/{total}** ({len(c['text'])} chars) → `{preview}...`")
        log.markdown("\n\n".join(lines[-8:]))  # keep the log short
        bar.progress(int(((i + 1) / total) * 100), text=f"Creating chunk {i + 1}/{total}")
        time.sleep(0.03)

    bar.progress(100, text=f"Done — created {total} chunks.")

    final_chunks = []
    for i, c in enumerate(raw_chunks):
        final_chunks.append({"id": i, "text": c["text"], "start": c["start"], "end": c["end"], "n_chars": len(c["text"])})
    return final_chunks


# ===========================================================================
# Retrieval (TF-IDF, runs fully offline — no API needed for search itself)
# ===========================================================================
def build_index(chunks):
    logger.info("Building TF-IDF index for %s chunks", len(chunks))
    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def retrieve(query, vectorizer, matrix, chunks, top_k=4):
    logger.info("Retrieving top %s chunks for query: %s", top_k, query[:120].replace('\n',' '))
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix).flatten()
    order = np.argsort(sims)[::-1][:top_k]
    results = []
    for idx in order:
        results.append({**chunks[idx], "score": float(sims[idx])})
    return results


def build_olama_prompt(system_prompt, question, retrieved_chunks, max_context_chars=3000, max_chunks=4):
    selected = retrieved_chunks[:max_chunks]
    parts = []
    total = 0
    truncated = False
    for c in selected:
        chunk_text = c["text"].strip()
        if total + len(chunk_text) > max_context_chars:
            remaining = max_context_chars - total
            if remaining <= 0:
                truncated = True
                break
            chunk_text = chunk_text[:remaining]
            truncated = True
        parts.append(f"[Chunk {c['id']} | relevance {c['score']:.2f}]\n{chunk_text}")
        total += len(chunk_text)
        if total >= max_context_chars:
            truncated = truncated or len(selected) < len(retrieved_chunks)
            break
    if len(selected) < len(retrieved_chunks):
        truncated = True
    context = "\n\n---\n\n".join(parts)
    prompt = (
        f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {question}\n\n"
        "Please answer using only the provided context. If the answer is not contained in the context, "
        "respond with 'I don't know.'"
    )
    logger.debug("Built Olama prompt with %s chunks and %s chars (truncated=%s)", len(parts), len(prompt), truncated)
    return prompt, truncated


# ===========================================================================
# Answer generation
# ===========================================================================
def generate_answer(question, retrieved_chunks, api_key, model, olama_url=None):
    context = "\n\n---\n\n".join(
        f"[Chunk {c['id']} | relevance {c['score']:.2f}]\n{c['text']}" for c in retrieved_chunks
    )
    system_prompt = (
        "You are a precise question-answering assistant. Answer the user's question "
        "using ONLY the provided context chunks. If the answer isn't in the context, "
        "say you don't have enough information. Cite chunk numbers you used, like [Chunk 2]."
    )
    user_message = f"Context:\n{context}\n\nQuestion: {question}"
    logger.info("Generating answer with model %s", model)
    logger.debug("Prompt length: %s", len(user_message))

    if model.startswith("claude"):
        if anthropic is None:
            return "The `anthropic` package isn't installed. Run: pip install anthropic"
        if not api_key:
            return "Enter an Anthropic API key in the sidebar to generate an answer."

        try:
            client = anthropic.Anthropic(api_key=api_key)
            resp = client.messages.create(
                model=model,
                max_tokens=1000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            parts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
            return "\n".join(parts) if parts else "(no text returned)"
        except Exception as e:
            return f"Error calling Anthropic API: {e}"

    if model == "gemma3-4b":
        if not olama_url:
            return "Enter your local Olama server URL in the sidebar to generate an answer."
        olama_model_id = "gemma3:4b"
        prompt, truncated = build_olama_prompt(system_prompt, question, retrieved_chunks)
        payload = {
            "model": olama_model_id,
            "prompt": prompt,
            "max_tokens": 512,
            "temperature": 0.0,
            "top_p": 1.0,
        }
        try:
            response = requests.post(
                f"{olama_url.rstrip('/')}/v1/completions",
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict):
                if "choices" in data and data["choices"]:
                    first = data["choices"][0]
                    if isinstance(first, dict):
                        if "text" in first:
                            answer_text = first["text"]
                        else:
                            msg = first.get("message")
                            answer_text = msg.get("content") if isinstance(msg, dict) else str(first)
                    else:
                        answer_text = str(first)
                else:
                    answer_text = data.get("output") or data.get("text") or str(data)
            else:
                answer_text = str(data)
            if truncated:
                warning = (
                    "**Note:** The local Olama prompt was truncated to fit input limits. "
                    "Some context may have been omitted from the answer."
                )
                return f"{warning}\n\n{answer_text}"
            return answer_text
        except Exception as e:
            logger.error("Local Olama request failed: %s", e, exc_info=True)
            response = getattr(e, 'response', None)
            return (
                f"Error calling local Olama server: {e}\n"
                f"URL: {olama_url.rstrip('/')}/v1/completions\n"
                f"Model: {olama_model_id}\n"
                f"Payload: {json.dumps(payload, ensure_ascii=False)}\n"
                f"Response status: {response.status_code if response is not None else 'N/A'}\n"
                f"Response text: {response.text if response is not None else 'N/A'}"
            )

    return f"Model `{model}` is not supported yet."

def read_log_lines(path, max_lines=30):
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return lines[-max_lines:]
    except FileNotFoundError:
        return ["Log file not found."]
    except Exception as e:
        logger.error("Error reading log file: %s", e, exc_info=True)
        return [f"Error reading log file: {e}"]

# ===========================================================================
# Sidebar — ingestion + chunking controls
# ===========================================================================
st.sidebar.title("⚙️ Setup")

uploaded_file = st.sidebar.file_uploader(
    "Upload a document", type=["json", "md", "markdown", "pdf"],
    help="PDF gets text-extracted first. JSON/Markdown are chunked directly.",
)

strategy = st.sidebar.radio("Chunking strategy", ["Fixed-size (characters)", "Paragraph-based"])
chunk_size = st.sidebar.slider("Chunk size (characters)", 200, 3000, 800, step=100)
overlap = st.sidebar.slider(
    "Overlap (characters)", 0, 500, 100, step=25,
    disabled=(strategy != "Fixed-size (characters)"),
)

process_clicked = st.sidebar.button("🚀 Process document", use_container_width=True, disabled=uploaded_file is None)

st.sidebar.divider()
st.sidebar.subheader("🤖 QA settings")
model = st.sidebar.selectbox(
    "Model",
    [
        "claude-sonnet-5",
        "claude-opus-4-8",
        "claude-haiku-4-5-20251001",
        "gemma3-4b",
        "gemini",
        "grok",
    ],
)
api_key = None
olama_url = None
if model.startswith("claude"):
    api_key = st.sidebar.text_input("Anthropic API key", type="password")
elif model == "gemma3-4b":
    st.sidebar.info("Using a local Olama server for gemma3-4b. No Anthropic key is needed.")
    olama_url = st.sidebar.text_input("Olama server URL", value="http://127.0.0.1:11434")
else:
    api_key = st.sidebar.text_input("Anthropic API key", type="password")

top_k = st.sidebar.slider("Chunks to retrieve", 1, 10, 4)

st.sidebar.divider()
with st.sidebar.expander("📝 App logs", expanded=False):
    log_lines = read_log_lines(LOG_DIR / "app.log", max_lines=50)
    st.code("".join(log_lines), language="text")


# ===========================================================================
# Main area
# ===========================================================================
st.title("🔎 RAG Playground")
st.caption("Upload → Parse (PDF only) → Chunk → Retrieve → Answer")

tab_process, tab_chunks, tab_qa = st.tabs(["📥 Processing", "🧩 Chunks", "💬 Question Answering"])

# ---- Processing tab -------------------------------------------------------
with tab_process:
    if uploaded_file is None:
        st.info("Upload a JSON, Markdown, or PDF file in the sidebar to get started.")
    elif process_clicked:
        ext = uploaded_file.name.split(".")[-1].lower()
        logger.info("Processing uploaded file %s with extension %s", uploaded_file.name, ext)
        file_bytes = uploaded_file.getvalue()

        if ext == "pdf":
            st.subheader("Step 1 — Parsing PDF")
            parse_container = st.container()
            with st.spinner("Extracting text from PDF..."):
                text = extract_text_from_pdf(file_bytes, parse_container)
            source_type = "pdf"
            with st.expander("Preview extracted text"):
                st.text(text[:3000] + ("..." if len(text) > 3000 else ""))
        elif ext == "json":
            st.subheader("Step 1 — Loading JSON")
            try:
                text = json_to_text(file_bytes)
                st.success("JSON loaded — chunking directly (no separate parse step needed).")
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")
                text = ""
            source_type = "json"
            with st.expander("Preview JSON as text"):
                st.text(text[:3000] + ("..." if len(text) > 3000 else ""))
        else:
            st.subheader("Step 1 — Loading Markdown")
            text = md_to_text(file_bytes)
            source_type = "markdown"
            st.success("Markdown loaded — chunking directly (no separate parse step needed).")
            with st.expander("Preview markdown text"):
                st.text(text[:3000] + ("..." if len(text) > 3000 else ""))

        st.session_state.raw_text = text
        st.session_state.source_type = source_type

        if text.strip():
            st.subheader("Step 2 — Chunking")
            viz_container = st.container()
            chunks = run_chunking_with_visualization(text, strategy, chunk_size, overlap, viz_container)
            st.session_state.chunks = chunks

            st.subheader("Step 3 — Building retrieval index")
            with st.spinner("Building TF-IDF index over chunks..."):
                vectorizer, matrix = build_index(chunks)
            st.session_state.vectorizer = vectorizer
            st.session_state.tfidf_matrix = matrix
            st.success(f"Ready! {len(chunks)} chunks indexed. Go to the 'Chunks' or 'Question Answering' tab.")
        else:
            st.warning("No text extracted — nothing to chunk.")
    elif st.session_state.chunks:
        st.info("Document already processed. See the 'Chunks' tab, or click 'Process document' again to redo it.")
    else:
        st.info("Click **Process document** in the sidebar to run parsing + chunking.")

# ---- Chunks tab ------------------------------------------------------------
with tab_chunks:
    chunks = st.session_state.chunks
    if not chunks:
        st.info("No chunks yet — process a document first.")
    else:
        st.write(f"**{len(chunks)} chunks** from a `{st.session_state.source_type}` source.")
        search_filter = st.text_input("Filter chunks by keyword (optional)")
        for c in chunks:
            if search_filter and search_filter.lower() not in c["text"].lower():
                continue
            with st.expander(f"Chunk {c['id']} — {c['n_chars']} chars (offset {c['start']}–{c['end']})"):
                st.text(c["text"])

# ---- QA tab -----------------------------------------------------------------
with tab_qa:
    chunks = st.session_state.chunks
    if not chunks:
        st.info("Process a document first to enable question answering.")
    else:
        question = st.text_input("Ask a question about the document")
        col1, col2 = st.columns([1, 1])
        ask_clicked = col1.button("Ask", type="primary", disabled=not question)
        col2.caption(f"Retrieving top {top_k} chunks via TF-IDF similarity, then asking {model}.")

        if ask_clicked and question:
            with st.spinner("Retrieving relevant chunks..."):
                retrieved = retrieve(
                    question, st.session_state.vectorizer, st.session_state.tfidf_matrix, chunks, top_k=top_k
                )

            st.subheader("📎 Retrieved chunks")
            for r in retrieved:
                with st.expander(f"Chunk {r['id']} — relevance {r['score']:.3f}"):
                    st.text(r["text"])

            st.subheader("🧠 Answer")
            logger.info("Asking question to model %s", model)
            with st.spinner("Generating answer..."):
                answer = generate_answer(question, retrieved, api_key, model, olama_url=olama_url)
            logger.info("Generated answer length %s", len(answer))
            st.markdown(answer)

            st.session_state.qa_history.append({"question": question, "answer": answer})

        if st.session_state.qa_history:
            with st.expander("📜 Q&A history"):
                for item in reversed(st.session_state.qa_history):
                    st.markdown(f"**Q:** {item['question']}")
                    st.markdown(f"**A:** {item['answer']}")
                    st.divider()
