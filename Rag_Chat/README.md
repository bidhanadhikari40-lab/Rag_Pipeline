# RAG Playground (Streamlit)

Upload a **JSON**, **Markdown**, or **PDF** file, watch it get parsed and chunked
step-by-step, browse the resulting chunks, and ask questions answered against
the chunks that are actually retrieved.

## How it works

1. **Upload** — sidebar file uploader accepts `.json`, `.md`, `.markdown`, `.pdf`.
2. **Parse (PDF only)** — PDFs are text-extracted page by page with a visible
   progress bar. JSON and Markdown skip this step and go straight to chunking
   (JSON is pretty-printed to text first so it's readable).
3. **Chunk** — choose fixed-size (with overlap) or paragraph-based chunking.
   The app replays the chunk-creation process live so you can watch chunks
   being built one at a time.
4. **Index** — chunks are indexed with TF-IDF (fully offline, no API key
   needed for search).
5. **Ask** — type a question. The app retrieves the top-k most relevant
   chunks (shown with relevance scores), then sends them as context to a
   Claude model to generate a grounded answer, citing chunk numbers.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Or on Windows, from the repository root:

```powershell
.\run_app.ps1
```

Enter your Anthropic API key in the sidebar under "QA settings" to enable
answer generation (retrieval and chunking work without a key).

## Notes

- TF-IDF retrieval is intentionally simple/offline — swap in embeddings
  (e.g. `voyageai` or a local sentence-transformers model) in `build_index`
  and `retrieve` if you want semantic search instead of keyword-based search.
- The chunking visualization is capped to the last 8 log lines on screen for
  readability, but every chunk is still created and available in the
  "Chunks" tab.
