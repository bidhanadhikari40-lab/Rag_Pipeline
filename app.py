"""
RAG Chatbot — main Streamlit application.

Flow:
  1. User logs in or registers (local JSON-backed auth).
  2. User uploads PDFs -> parsed via LlamaParse.
  3. Parsed text is chunked.
  4. Chunks are embedded (Sentence-Transformers) and indexed (FAISS).
  5. User asks a question -> top chunks retrieved -> answer generated
     by a local Ollama model (gemma3), grounded in the retrieved context.
"""

import os
import subprocess
import sys

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("LLAMA_CLOUD_API_KEY"):
    os.environ["LLAMA_CLOUD_API_KEY"] = os.getenv("LLAMA_PARSE_KEY") or os.getenv("llama_parse_key") or ""

from auth import authenticate_user, register_user
from pdf_parser import parse_pdf_bytes
from chunking import chunk_text
from vector_store import VectorStore
from generation import generate_answer_stream, get_model_status

if __name__ == "__main__" and os.getenv("DOC_LAB_STREAMLIT_LAUNCHED") != "1":
    is_streamlit_invocation = any(arg.startswith("--") for arg in sys.argv[1:]) or any(arg == "run" for arg in sys.argv[1:])
    if not is_streamlit_invocation:
        app_path = os.path.abspath(__file__)
        project_dir = os.path.dirname(app_path)
        env = os.environ.copy()
        env["DOC_LAB_STREAMLIT_LAUNCHED"] = "1"
        command = [sys.executable, "-m", "streamlit", "run", app_path, "--server.headless", "true", "--server.port", "8501"]
        print("Launching DocLab with Streamlit...")
        subprocess.Popen(command, cwd=project_dir, env=env)
        print("Open http://localhost:8501")
        sys.exit(0)

st.set_page_config(page_title="DocLab — RAG Assistant", page_icon="📄", layout="wide")

# ----------------------------------------------------------------------
# Styling and Themes
# ----------------------------------------------------------------------
THEME_PALETTES = {
    "Midnight": {
        "bg": "#14171c",
        "panel": "#1b1f27",
        "panel_border": "#2a2f3a",
        "text": "#e7e9ee",
        "text_dim": "#8b919e",
        "accent": "#d98c4a",
        "accent_soft": "rgba(217, 140, 74, 0.14)",
    },
    "Ocean": {
        "bg": "#071922",
        "panel": "#102635",
        "panel_border": "#214a5f",
        "text": "#eef8ff",
        "text_dim": "#8ab0c0",
        "accent": "#43b8c8",
        "accent_soft": "rgba(67, 184, 200, 0.15)",
    },
    "Forest": {
        "bg": "#0c1c16",
        "panel": "#142a21",
        "panel_border": "#2b4939",
        "text": "#f1f8f2",
        "text_dim": "#8fb29b",
        "accent": "#4fbf7d",
        "accent_soft": "rgba(79, 191, 125, 0.16)",
    },
    "Rose": {
        "bg": "#23131b",
        "panel": "#34202b",
        "panel_border": "#5b3542",
        "text": "#fff2f7",
        "text_dim": "#d8b4c1",
        "accent": "#ff7ea8",
        "accent_soft": "rgba(255, 126, 168, 0.15)",
    },
}


def apply_theme(theme_name: str):
    palette = THEME_PALETTES.get(theme_name, THEME_PALETTES["Midnight"])
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

        :root {{
            --bg: {palette['bg']};
            --panel: {palette['panel']};
            --panel-border: {palette['panel_border']};
            --text: {palette['text']};
            --text-dim: {palette['text_dim']};
            --accent: {palette['accent']};
            --accent-soft: {palette['accent_soft']};
            --mono: 'JetBrains Mono', monospace;
        }}

        .stApp {{ background: linear-gradient(135deg, var(--bg), color-mix(in srgb, var(--bg) 75%, black 25%)); color: var(--text); }}
        h1, h2, h3 {{ font-family: 'Source Serif 4', serif !important; letter-spacing: -0.01em; }}

        .app-shell {{ max-width: 1320px; margin: 0 auto; padding: 0.4rem 0 2rem; }}
        .hero-banner {{ display: grid; grid-template-columns: 1.2fr 0.8fr; gap: 18px; padding: 30px; border: 1px solid var(--panel-border); border-radius: 28px; background: linear-gradient(135deg, color-mix(in srgb, var(--panel) 92%, white 8%), color-mix(in srgb, var(--panel) 80%, var(--accent) 20%)); box-shadow: 0 22px 55px rgba(0,0,0,0.22); margin-bottom: 18px; }}
        .hero-copy {{ display: flex; flex-direction: column; gap: 10px; }}
        .hero-title {{ font-size: 2.15rem; margin: 0; line-height: 1.15; }}
        .hero-subtitle {{ color: var(--text-dim); font-size: 1rem; line-height: 1.7; margin: 0; max-width: 720px; }}
        .hero-pill {{ display:inline-flex; align-items:center; gap:8px; padding:6px 10px; border-radius:999px; background: var(--accent-soft); color: var(--accent); font-size:12px; font-family: var(--mono); width: fit-content; }}
        .section-card {{ background: rgba(255,255,255,0.035); border: 1px solid var(--panel-border); border-radius: 20px; padding: 18px 20px; margin-bottom: 12px; box-shadow: 0 14px 30px rgba(0,0,0,0.13); backdrop-filter: blur(8px); }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-top: 12px; }}
        .metric-card {{ background: color-mix(in srgb, var(--panel) 90%, white 10%); border: 1px solid var(--panel-border); border-radius: 16px; padding: 14px; }}
        .metric-value {{ font-size: 1.15rem; font-weight: 700; color: var(--text); }}
        .metric-label {{ font-size: 0.9rem; color: var(--text-dim); margin-top: 4px; }}
        .pipeline {{ display: flex; align-items: center; gap: 0; margin: 4px 0 28px 0; flex-wrap: wrap; }}
        .pipeline-step {{ display: flex; align-items: center; gap: 8px; padding: 6px 14px 6px 10px; border-radius: 999px; font-family: var(--mono); font-size: 12.5px; color: var(--text-dim); border: 1px solid var(--panel-border); background: var(--panel); }}
        .pipeline-step.active {{ color: var(--accent); border-color: var(--accent); background: var(--accent-soft); }}
        .pipeline-step .dot {{ width: 6px; height: 6px; border-radius: 50%; background: var(--text-dim); }}
        .pipeline-step.active .dot {{ background: var(--accent); }}
        .pipeline-arrow {{ color: var(--panel-border); font-size: 14px; margin: 0 6px; }}
        .doclab-card {{ background: rgba(255,255,255,0.035); border: 1px solid var(--panel-border); border-radius: 14px; padding: 16px 18px; margin-bottom: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.1); }}
        .chunk-meta {{ font-family: var(--mono); font-size: 11.5px; color: var(--accent); margin-bottom: 4px; }}
        .stButton button {{ border-radius: 10px; font-weight: 600; box-shadow: 0 8px 18px rgba(0,0,0,0.15); transition: transform 0.2s ease; }}
        .stButton button:hover {{ transform: translateY(-1px); }}
        section[data-testid="stSidebar"] {{ background-color: color-mix(in srgb, var(--bg) 92%, black 8%); border-right: 1px solid var(--panel-border); }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------
# Session State Initialization
# ----------------------------------------------------------------------
def init_state():
    defaults = {
        "logged_in": False,
        "username": None,
        "auth_mode": "Login",
        "theme": "Midnight",
        "parsed_docs": {},       # filename -> parsed text
        "chunks_by_file": {},    # filename -> list[Chunk]
        "all_chunks": [],        # flat list[Chunk] across all files
        "selected_docs_for_chat": [],
        "vector_store": VectorStore(),
        "chat_history": [],      # list of {"role", "content", "retrieved": [...]}
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "auth_transition" not in st.session_state:
        st.session_state["auth_transition"] = False

    if st.session_state.get("auth_transition"):
        st.session_state["auth_transition"] = False

    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if "username" not in st.session_state:
        st.session_state["username"] = None


init_state()
apply_theme(st.session_state.get("theme", "Midnight"))


# ----------------------------------------------------------------------
# Auth Screen
# ----------------------------------------------------------------------
def render_auth_screen():
    apply_theme(st.session_state.get("theme", "Midnight"))

    st.markdown("<div class='app-shell'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='hero-banner'>
          <div class='hero-copy'>
            <div class='hero-pill'>⚡ Private RAG Workspace</div>
            <h1 class='hero-title'>DocLab turns your documents into a smart, local assistant.</h1>
            <p class='hero-subtitle'>Upload PDFs, parse them with LlamaParse, index the content locally, and ask questions with transparent retrieval and a private Ollama-backed answer engine.</p>
          </div>
          <div class='section-card'>
            <div class='chunk-meta'>PICK A THEME</div>
            <p style='margin: 6px 0 12px; color: var(--text-dim);'>Choose a background that fits your mood while you work.</p>
            <div style='margin-bottom: 10px;'>
            """ + f"<div class='hero-pill'>🎨 {st.session_state.get('theme', 'Midnight')}</div>" + """
            </div>
            <div style='margin-top: 10px;'>
            """
        , unsafe_allow_html=True,
    )
    theme_choice = st.selectbox("Background theme", list(THEME_PALETTES.keys()), index=list(THEME_PALETTES.keys()).index(st.session_state.get("theme", "Midnight")), key="auth_theme")
    st.session_state.theme = theme_choice
    apply_theme(st.session_state.theme)
    st.markdown(
        """
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<h2 style='margin-top:0;'>Welcome back</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color:var(--text-dim); line-height:1.6;'>Sign in to continue your workspace or create a fresh account to start exploring offline.</p>", unsafe_allow_html=True)
        st.markdown("<div class='doclab-card' style='margin-top:14px;'>", unsafe_allow_html=True)
        mode = st.radio("Account", ["Login", "Register"], horizontal=True, label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

        if mode == "Login":
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Log in", use_container_width=True)

            if submitted:
                ok, message = authenticate_user(username, password)
                if ok:
                    st.session_state.logged_in = True
                    st.session_state.username = username.strip().lower()
                    st.session_state["auth_transition"] = True
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

        else:
            with st.form("register_form"):
                new_username = st.text_input("Choose a username")
                new_email = st.text_input("Email (optional)")
                new_password = st.text_input("Choose a password", type="password")
                confirm_password = st.text_input("Confirm password", type="password")
                submitted = st.form_submit_button("Create account", use_container_width=True)

            if submitted:
                if new_password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    ok, message = register_user(new_username, new_password, new_email)
                    if ok:
                        st.success(message + " Switch to the Login tab to continue.")
                    else:
                        st.error(message)

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(
            """
            <div class='section-card'>
            <h3 style='margin-top:0;'>What this experience feels like</h3>
            <p style='color:var(--text-dim); line-height:1.6;'>
            Each step is visible and lightweight: upload, parse, chunk, search, and answer — all within a streamlined interface.
            </p>
            <div class='metric-grid'>
              <div class='metric-card'>
                <div class='metric-value'>Local</div>
                <div class='metric-label'>No cloud dependency for answering</div>
              </div>
              <div class='metric-card'>
                <div class='metric-value'>Transparent</div>
                <div class='metric-label'>See retrieved chunks per answer</div>
              </div>
              <div class='metric-card'>
                <div class='metric-value'>Fast</div>
                <div class='metric-label'>Go from PDF to insight quickly</div>
              </div>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Pipeline Stepper
# ----------------------------------------------------------------------
def render_pipeline(active_step: str):
    steps = ["Upload", "Parse", "Chunk", "Retrieve", "Ask"]
    html = "<div class='pipeline'>"
    for i, step in enumerate(steps):
        is_active = step == active_step
        cls = "pipeline-step active" if is_active else "pipeline-step"
        html += f"<div class='{cls}'><span class='dot'></span>{step}</div>"
        if i < len(steps) - 1:
            html += "<span class='pipeline-arrow'>→</span>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Main App (post-login)
# ----------------------------------------------------------------------
def render_main_app():
    with st.sidebar:
        st.markdown(f"**Signed in as**  \n`{st.session_state.username}`")
        if st.button("Log out", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.rerun()

        st.divider()
        st.markdown("**Appearance**")
        theme_choice = st.selectbox("Background", list(THEME_PALETTES.keys()), index=list(THEME_PALETTES.keys()).index(st.session_state.get("theme", "Midnight")), key="theme_selector")
        st.session_state.theme = theme_choice
        apply_theme(st.session_state.theme)
        st.divider()
        st.markdown("**Indexed documents**")
        if st.session_state.parsed_docs:
            for fname in st.session_state.parsed_docs:
                n_chunks = len(st.session_state.chunks_by_file.get(fname, []))
                st.markdown(f"- `{fname}`  \n  <span class='chunk-meta'>{n_chunks} chunks</span>", unsafe_allow_html=True)
        else:
            st.caption("No documents uploaded yet.")

        st.divider()
        with st.expander("Chunking settings"):
            st.session_state.setdefault("chunk_size", 800)
            st.session_state.setdefault("chunk_overlap", 150)
            st.session_state.chunk_size = st.slider("Chunk size (characters)", 300, 2000, st.session_state.chunk_size, 50)
            st.session_state.chunk_overlap = st.slider("Chunk overlap (characters)", 0, 400, st.session_state.chunk_overlap, 25)
        with st.expander("Retrieval settings"):
            st.session_state.setdefault("top_k", 4)
            st.session_state.top_k = st.slider("Chunks retrieved per question", 1, 10, st.session_state.top_k)

    st.markdown("<div class='app-shell'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='hero-banner'>
          <div class='hero-copy'>
            <div class='hero-pill'>🧠 Local-first RAG</div>
            <h1 class='hero-title'>Your documents, now searchable and conversational.</h1>
            <p class='hero-subtitle'>Upload one or more PDFs, parse them, build a local index, and ask grounded questions with evidence-backed answers.</p>
          </div>
          <div class='section-card'>
            <div class='chunk-meta'>STATUS</div>
            <div style='display:flex; flex-direction:column; gap:10px;'>
        """,
        unsafe_allow_html=True,
    )
    model_ready, model_message = get_model_status()
    if model_ready:
        st.success("🟢 Local model ready via Ollama")
    else:
        st.warning(f"⚠️ Local model check: {model_message}")
    st.markdown(
        """
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='metric-grid'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='metric-card'><div class='metric-value'>Upload</div><div class='metric-label'>PDFs to parse</div></div>
        <div class='metric-card'><div class='metric-value'>Chunk</div><div class='metric-label'>Build grounded context</div></div>
        <div class='metric-card'><div class='metric-value'>Ask</div><div class='metric-label'>Get answers with evidence</div></div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    tab_upload, tab_chunks, tab_chat = st.tabs(["📤 Upload & Parse", "🧩 Chunks", "💬 Ask"])

    # ---------------- Tab 1: Upload & Parse ----------------
    with tab_upload:
        render_pipeline("Parse" if st.session_state.parsed_docs else "Upload")

        uploaded_files = st.file_uploader(
            "Upload one or more PDF files", type=["pdf"], accept_multiple_files=True
        )

        if uploaded_files and st.button("Parse uploaded PDFs", type="primary"):
            progress = st.progress(0.0, text="Starting...")
            total = len(uploaded_files)

            for i, uploaded_file in enumerate(uploaded_files):
                progress.progress(i / total, text=f"Parsing {uploaded_file.name} ...")
                try:
                    text = parse_pdf_bytes(uploaded_file.getvalue(), uploaded_file.name)
                    st.session_state.parsed_docs[uploaded_file.name] = text
                except Exception as error:
                    st.error(f"Failed to parse {uploaded_file.name}: {error}")
            progress.progress(1.0, text="Done.")
            st.success(f"Parsed {total} file(s). View the content below, or move to the Chunks tab.")

        if st.session_state.parsed_docs:
            st.markdown("#### Parsed content")
            chosen_file = st.selectbox("Select a document to preview", list(st.session_state.parsed_docs.keys()))
            if chosen_file:
                st.markdown("<div class='doclab-card'>", unsafe_allow_html=True)
                st.text_area(
                    "Parsed text",
                    st.session_state.parsed_docs[chosen_file],
                    height=350,
                    label_visibility="collapsed",
                )
                st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- Tab 2: Chunks ----------------
    with tab_chunks:
        render_pipeline("Chunk")

        if not st.session_state.parsed_docs:
            st.info("Upload and parse at least one PDF first.")
        else:
            if st.button("Generate chunks for all documents", type="primary"):
                st.session_state.chunks_by_file = {}
                all_chunks = []
                for fname, text in st.session_state.parsed_docs.items():
                    chunks = chunk_text(
                        text,
                        fname,
                        chunk_size=st.session_state.chunk_size,
                        chunk_overlap=st.session_state.chunk_overlap,
                    )
                    st.session_state.chunks_by_file[fname] = chunks
                    all_chunks.extend(chunks)
                st.session_state.all_chunks = all_chunks

                with st.spinner("Embedding chunks and building the search index..."):
                    st.session_state.vector_store.build(all_chunks)

                st.success(f"Generated {len(all_chunks)} chunks and built the vector index.")

            if st.session_state.chunks_by_file:
                st.markdown("#### Generated chunks")
                chosen_file = st.selectbox(
                    "Filter by document", list(st.session_state.chunks_by_file.keys()), key="chunk_view_select"
                )
                chunks = st.session_state.chunks_by_file.get(chosen_file, [])
                st.caption(f"{len(chunks)} chunks from `{chosen_file}`")

                for chunk in chunks:
                    st.markdown(
                        f"""<div class='doclab-card'>
                        <div class='chunk-meta'>{chunk.chunk_id}</div>
                        {chunk.text}
                        </div>""",
                        unsafe_allow_html=True,
                    )

    # ---------------- Tab 3: Ask ----------------
    with tab_chat:
        render_pipeline("Retrieve" if not st.session_state.chat_history else "Ask")

        if not st.session_state.vector_store.is_ready():
            st.info("Upload PDFs and generate chunks first (see the previous tabs).")
        else:
            for entry in st.session_state.chat_history:
                with st.chat_message(entry["role"]):
                    st.markdown(entry["content"])
                    if entry["role"] == "assistant" and entry.get("retrieved"):
                        with st.expander("Retrieved chunks used for this answer"):
                            for chunk in entry["retrieved"]:
                                st.markdown(
                                    f"""<div class='doclab-card'>
                                    <div class='chunk-meta'>{chunk['source_file']} · chunk #{chunk['chunk_index']} · score {chunk['score']:.3f}</div>
                                    {chunk['text']}
                                    </div>""",
                                    unsafe_allow_html=True,
                                )

            doc_options = list(st.session_state.parsed_docs.keys())
            if doc_options:
                selected_docs = st.multiselect(
                    "Use these documents for retrieval",
                    options=doc_options,
                    default=doc_options if not st.session_state.get("selected_docs_for_chat") else st.session_state.selected_docs_for_chat,
                    key="selected_docs_for_chat",
                )
                st.session_state.selected_docs_for_chat = selected_docs
            else:
                st.session_state.selected_docs_for_chat = []

            question = st.chat_input("Ask a question about your uploaded documents...")

            if question:
                st.session_state.chat_history.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.markdown(question)

                retrieved = st.session_state.vector_store.search(question, top_k=st.session_state.top_k)
                selected_docs = set(st.session_state.get("selected_docs_for_chat", []))
                if selected_docs:
                    retrieved = [chunk for chunk in retrieved if chunk.get("source_file") in selected_docs]

                with st.chat_message("assistant"):
                    placeholder = st.empty()
                    answer = ""
                    try:
                        for token in generate_answer_stream(question, retrieved):
                            answer += token
                            placeholder.markdown(answer + "▌")
                        placeholder.markdown(answer)
                    except Exception as error:
                        answer = f"Sorry, the local model returned an error: {error}"
                        placeholder.markdown(answer)

                    if retrieved:
                        with st.expander("Retrieved chunks used for this answer"):
                            for chunk in retrieved:
                                st.markdown(
                                    f"""<div class='doclab-card'>
                                    <div class='chunk-meta'>{chunk['source_file']} · chunk #{chunk['chunk_index']} · score {chunk['score']:.3f}</div>
                                    {chunk['text']}
                                    </div>""",
                                    unsafe_allow_html=True,
                                )

                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer, "retrieved": retrieved}
                )


# ----------------------------------------------------------------------
# Router
# ----------------------------------------------------------------------
if st.session_state.logged_in:
    render_main_app()
else:
    render_auth_screen()
