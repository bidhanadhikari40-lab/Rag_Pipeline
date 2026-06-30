# DocLab — RAG Chatbot

A Streamlit-based RAG (Retrieval-Augmented Generation) chatbot with login/register,
PDF parsing via LlamaParse, local chunking, local embeddings (Sentence-Transformers),
local vector search (FAISS), and local answer generation (Ollama + gemma3).

## Project structure

```
ragchatbot/
├── app.py             # Main Streamlit app (UI, routing, auth gate)
├── auth.py            # Login/register, JSON-backed user store
├── pdf_parser.py       # LlamaParse integration
├── chunking.py         # Text chunking logic
├── vector_store.py     # Sentence-Transformers embeddings + FAISS index
├── generation.py       # Ollama (gemma3) answer generation
├── requirements.txt
└── users.json           # Created automatically on first registration
```

## 1. Prerequisites

### a) Python packages
```bash
pip install -r requirements.txt
```

### b) Ollama + gemma3 (local LLM for answer generation)
1. Install Ollama: https://ollama.com/download
2. Pull the model:
   ```bash
   ollama pull gemma3
   ```
3. Make sure the Ollama app/service is running in the background before
   starting the Streamlit app.

### c) LlamaParse API key (for PDF parsing)
1. Get a free API key at https://cloud.llamaindex.ai
2. Set it as an environment variable before running the app:

   **Windows (PowerShell):**
   ```powershell
   $env:LLAMA_CLOUD_API_KEY="your-key-here"
   ```

   **macOS/Linux:**
   ```bash
   export LLAMA_CLOUD_API_KEY="your-key-here"
   ```

   Or create a `.env` file in the project root with:
   ```
   LLAMA_CLOUD_API_KEY=your-key-here
   ```
   and load it with `python-dotenv` (add `from dotenv import load_dotenv; load_dotenv()`
   at the top of `app.py` if you go this route — not included by default to keep
   the dependency list minimal).

## 2. Run the app

```bash
streamlit run app.py
```

## 3. Using the app

1. **Register** a new account, then **log in**.
2. Go to **Upload & Parse**: upload one or more PDFs, click "Parse uploaded PDFs",
   and preview the parsed text per file.
3. Go to **Chunks**: click "Generate chunks for all documents" to split the parsed
   text into overlapping chunks and build the local FAISS vector index. Adjust
   chunk size/overlap in the sidebar before generating if needed.
4. Go to **Ask**: ask any question about your uploaded documents. Each answer
   shows an expandable "Retrieved chunks used for this answer" section so you
   can see exactly which passages were used to ground the response.

## Notes

- User credentials are stored in a local `users.json` file with salted
  SHA-256 password hashes — suitable for a class project/demo, **not**
  for production use.
- All embeddings, vector search, and generation run entirely on your machine;
  only the PDF parsing step calls out to the LlamaParse cloud API.
- The vector index is rebuilt in memory each session (not persisted to disk).
  Re-upload and re-chunk after restarting the app.
