import datetime
from pathlib import Path

import streamlit as st

from chatbot import OLLAMA_MODEL, build_answer
from database import (
    add_chat_message,
    authenticate,
    create_admin,
    create_chat_session,
    delete_chat_session,
    init_db,
    list_chat_sessions,
    load_chat_messages,
    load_pages_from_db,
    rename_chat_session,
    upsert_scraped_pages,
)
from scraper import DATA_PATH, load_pages, save_pages, scrape_site


st.set_page_config(
    page_title="LICT Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── ChatGPT-style CSS ────────────────────────────────────────────────────
CHATGPT_CSS = """
<style>
    /* Dark sidebar like ChatGPT */
    section[data-testid="stSidebar"] {
        background-color: #171717 !important;
        min-width: 280px !important;
    }
    section[data-testid="stSidebar"] * {
        color: #ececec !important;
    }
    
    /* Sidebar content padding */
    section[data-testid="stSidebar"] .block-container {
        padding: 0.75rem 0.75rem !important;
    }
    
    /* New Chat button - white outline style like ChatGPT */
    section[data-testid="stSidebar"] .stButton button[kind="primary"] {
        background: transparent !important;
        border: 1px solid #555 !important;
        color: #ececec !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        width: 100% !important;
        transition: background 0.15s !important;
    }
    section[data-testid="stSidebar"] .stButton button[kind="primary"]:hover {
        background: #2a2a2a !important;
        border-color: #777 !important;
    }
    
    /* Regular sidebar buttons (chat history items) */
    section[data-testid="stSidebar"] .stButton button {
        background: transparent !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 6px 10px !important;
        font-size: 0.82rem !important;
        text-align: left !important;
        width: 100% !important;
        color: #b4b4b4 !important;
        transition: background 0.15s !important;
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        background: #2a2a2a !important;
        color: #ececec !important;
    }
    
    /* Delete button (secondary) */
    section[data-testid="stSidebar"] .stButton button[kind="secondary"] {
        text-align: center !important;
        padding: 4px 4px !important;
        font-size: 0.75rem !important;
        color: #666 !important;
        min-width: 28px !important;
    }
    section[data-testid="stSidebar"] .stButton button[kind="secondary"]:hover {
        color: #ff4444 !important;
        background: #3a1a1a !important;
    }
    
    /* Divider */
    section[data-testid="stSidebar"] hr {
        border-color: #333 !important;
        margin: 12px 0 !important;
    }
    
    /* Subheader */
    section[data-testid="stSidebar"] h2 {
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        color: #888 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        margin: 8px 0 !important;
        padding: 0 4px !important;
    }
    
    /* Expander header */
    section[data-testid="stSidebar"] .streamlit-expanderHeader {
        font-size: 0.82rem !important;
        color: #b4b4b4 !important;
        background: transparent !important;
        border: none !important;
        padding: 6px 4px !important;
        border-radius: 6px !important;
    }
    section[data-testid="stSidebar"] .streamlit-expanderHeader:hover {
        background: #2a2a2a !important;
    }
    
    /* Expander content */
    section[data-testid="stSidebar"] .streamlit-expanderContent {
        border: none !important;
        padding: 4px 0 !important;
    }
    
    /* Selectbox */
    section[data-testid="stSidebar"] .stSelectbox label {
        color: #888 !important;
        font-size: 0.75rem !important;
    }
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
        background: #2a2a2a !important;
        border: 1px solid #444 !important;
        border-radius: 6px !important;
    }
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"]:hover {
        border-color: #666 !important;
    }
    
    /* Text input */
    section[data-testid="stSidebar"] .stTextInput label {
        color: #888 !important;
        font-size: 0.75rem !important;
    }
    section[data-testid="stSidebar"] .stTextInput input {
        background: #2a2a2a !important;
        border: 1px solid #444 !important;
        border-radius: 6px !important;
        color: #ececec !important;
    }
    section[data-testid="stSidebar"] .stTextInput input:focus {
        border-color: #666 !important;
    }
    
    /* Toggle */
    section[data-testid="stSidebar"] .stToggle label {
        color: #b4b4b4 !important;
        font-size: 0.82rem !important;
    }
    
    /* Slider */
    section[data-testid="stSidebar"] .stSlider label {
        color: #888 !important;
        font-size: 0.75rem !important;
    }
    
    /* Metric */
    section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #ececec !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: #888 !important;
        font-size: 0.75rem !important;
    }
    
    /* Caption */
    section[data-testid="stSidebar"] .stCaption {
        color: #666 !important;
        font-size: 0.7rem !important;
    }
    
    /* Info box */
    section[data-testid="stSidebar"] .stAlert {
        background: #2a2a2a !important;
        border: 1px solid #444 !important;
        border-radius: 6px !important;
        color: #b4b4b4 !important;
        font-size: 0.8rem !important;
        padding: 8px 12px !important;
    }
    
    /* Write text */
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #b4b4b4 !important;
        font-size: 0.82rem !important;
    }
    
    /* Main area - light theme like ChatGPT */
    .main {
        background-color: #ffffff !important;
    }
    .main .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 1.5rem !important;
        max-width: 800px !important;
        margin: 0 auto !important;
    }
    
    /* Chat messages */
    .stChatMessage {
        border-radius: 12px !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Chat input */
    .stChatInputContainer {
        border: 1px solid #ddd !important;
        border-radius: 12px !important;
        padding: 4px !important;
    }
    .stChatInputContainer:focus-within {
        border-color: #1a73e8 !important;
        box-shadow: 0 0 0 2px rgba(26,115,232,0.2) !important;
    }
</style>
"""


# ── Helper Functions ─────────────────────────────────────────────────────

def load_knowledge_base() -> list[dict]:
    db_pages = load_pages_from_db()
    if db_pages:
        return db_pages
    return load_pages(DATA_PATH)


def sync_json_to_db() -> int:
    pages = load_pages(DATA_PATH)
    if not pages:
        return 0
    return upsert_scraped_pages(pages)


def get_or_create_current_session() -> int:
    user = st.session_state.get("user")
    if not user:
        return 0
    session_id = st.session_state.get("current_session_id", 0)
    if not session_id:
        try:
            session_id = create_chat_session(user["id"])
            st.session_state.current_session_id = session_id
            st.session_state.messages = []
        except Exception:
            st.session_state.current_session_id = 0
            st.session_state.messages = []
    return session_id


def load_session_messages(session_id: int) -> list[dict]:
    user = st.session_state.get("user")
    if not user or not session_id:
        return []
    try:
        db_messages = load_chat_messages(user["id"], session_id)
        return [{"role": msg["role"], "content": msg["content"]} for msg in db_messages]
    except Exception:
        return []


def save_message_to_db(session_id: int, role: str, content: str) -> None:
    user = st.session_state.get("user")
    if not user or not session_id or not content:
        return
    try:
        add_chat_message(user["id"], session_id, role, content)
    except Exception:
        pass


def switch_session(session_id: int) -> None:
    st.session_state.current_session_id = session_id
    messages = load_session_messages(session_id)
    st.session_state.messages = messages if messages else []


def new_chat() -> None:
    user = st.session_state.get("user")
    if user:
        try:
            session_id = create_chat_session(user["id"])
            st.session_state.current_session_id = session_id
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "👋 Hello! I'm the LICT assistant. Ask me anything about courses, admissions, notices, faculty, or any other information from the LICT website.",
                }
            ]
            st.rerun()
        except Exception as e:
            st.error(f"Could not create new session: {e}")


def format_timestamp(ts: str) -> str:
    try:
        dt = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
        now = datetime.datetime.now(dt.tzinfo)
        diff = now - dt
        if diff.days == 0:
            return f"Today {dt.strftime('%I:%M %p')}"
        elif diff.days == 1:
            return f"Yesterday {dt.strftime('%I:%M %p')}"
        elif diff.days < 7:
            return dt.strftime('%A %I:%M %p')
        else:
            return dt.strftime('%b %d, %Y')
    except Exception:
        return ts[:10] if ts else ""


# ── Auth UI ──────────────────────────────────────────────────────────────

def render_auth() -> None:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
            <div style="text-align: center; padding: 2rem 0 1rem 0;">
                <h1 style="font-size: 2.5rem; margin-bottom: 0.3rem;">🎓 LICT Chatbot</h1>
                <p style="font-size: 1.1rem; color: #666; margin: 0;">
                    Your intelligent assistant for Lumbini ICT Campus information
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        login_tab, register_tab = st.tabs(["🔑 Sign In", "📝 Register"])
        with login_tab:
            with st.form("login_form"):
                st.text_input("Email", placeholder="your@email.com", key="login_email")
                st.text_input("Password", type="password", placeholder="Enter your password", key="login_password")
                submitted = st.form_submit_button("Sign in", use_container_width=True, type="primary")
            if submitted:
                email = st.session_state.get("login_email", "")
                password = st.session_state.get("login_password", "")
                user = authenticate(email, password)
                if user:
                    st.session_state.user = dict(user)
                    st.session_state.messages = []
                    st.session_state.current_session_id = 0
                    st.rerun()
                st.error("❌ Invalid email or password.")
        with register_tab:
            with st.form("register_form"):
                st.text_input("Full Name", placeholder="Your name", key="reg_name")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.text_input("Email", placeholder="your@email.com", key="reg_email")
                with col_b:
                    st.text_input("Password", type="password", placeholder="Min 6 chars", key="reg_password")
                st.text_input("Contact Number", placeholder="Phone number", key="reg_contact")
                st.text_area("Address", placeholder="Your address", height=80, key="reg_address")
                registered = st.form_submit_button("Create Account", use_container_width=True, type="primary")
            if registered:
                name = st.session_state.get("reg_name", "")
                contact = st.session_state.get("reg_contact", "")
                email = st.session_state.get("reg_email", "")
                password = st.session_state.get("reg_password", "")
                address = st.session_state.get("reg_address", "")
                if len(password) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    ok, message = create_admin(name, contact, email, password, address)
                    if ok:
                        st.success(f"✅ {message}")
                    else:
                        st.error(f"❌ {message}")


# ── Sidebar (ChatGPT Style) ─────────────────────────────────────────────

def render_sidebar() -> list[dict]:
    user = st.session_state["user"]
    
    # ── New Chat button (like ChatGPT's top button) ──
    if st.sidebar.button("+  New Chat", use_container_width=True, type="primary"):
        new_chat()
    
    st.sidebar.divider()
    
    # ── Chat History ──
    st.sidebar.subheader("Chat History")
    
    try:
        sessions = list_chat_sessions(user["id"])
    except Exception:
        sessions = []
    
    current_id = st.session_state.get("current_session_id", 0)
    
    if not sessions:
        st.sidebar.info("No conversations yet", icon="💡")
    else:
        for session in sessions[:20]:
            sid = session["id"]
            title = session.get("title", "New Chat")[:35]
            msg_count = session.get("message_count", 0)
            updated = format_timestamp(session.get("updated_at", ""))
            is_active = sid == current_id
            
            cols = st.sidebar.columns([5, 1])
            with cols[0]:
                label = f"{'▸ ' if is_active else '  '}{title}"
                if st.button(
                    label,
                    key=f"session_{sid}",
                    use_container_width=True,
                    help=f"{msg_count} messages • Last: {updated}",
                ):
                    switch_session(sid)
                    st.rerun()
            with cols[1]:
                if is_active:
                    if st.button("✕", key=f"del_{sid}", help="Delete this chat"):
                        try:
                            delete_chat_session(user["id"], sid)
                            if current_id == sid:
                                st.session_state.current_session_id = 0
                                st.session_state.messages = []
                            st.rerun()
                        except Exception as e:
                            st.error(f"Could not delete: {e}")
    
    st.sidebar.divider()

    # ── Settings (collapsible) ──
    with st.sidebar.expander("Settings", expanded=False):
        # Summary mode toggle
        st.toggle(
            "Auto-summarize mode",
            value=st.session_state.get("summarize_mode", True),
            key="summarize_toggle",
            help="When on, queries containing 'summarize', 'overview', etc. will automatically summarize the content"
        )
        st.session_state.summarize_mode = st.session_state.summarize_toggle

        model_type = st.selectbox(
            "AI Model",
            options=["Ollama (Local)", "Gemini (Google)", "Grok (xAI)"],
            index=0,
            key="model_type_select",
        )
        model_type_map = {
            "Ollama (Local)": "ollama",
            "Gemini (Google)": "gemini",
            "Grok (xAI)": "grok",
        }
        st.session_state.model_type = model_type_map[model_type]

        if st.session_state.model_type == "ollama":
            st.toggle("Enable AI", value=st.session_state.get("use_ai", True), key="ollama_toggle")
            st.session_state.use_ai = st.session_state.ollama_toggle
            model_name = st.text_input(
                "Model Name",
                value=st.session_state.get("ollama_model", OLLAMA_MODEL),
                key="ollama_model_name",
            )
            st.session_state.ollama_model = model_name
            st.caption("Requires Ollama on localhost:11434")
        elif st.session_state.model_type == "gemini":
            st.toggle("Enable Gemini", value=st.session_state.get("use_ai", False), key="gemini_toggle")
            st.session_state.use_ai = st.session_state.gemini_toggle
            api_key = st.text_input("API Key", type="password", value=st.session_state.get("gemini_api_key", ""), key="gemini_key")
            if api_key:
                st.session_state.gemini_api_key = api_key
            st.caption("Get key from ai.google.dev")
        elif st.session_state.model_type == "grok":
            st.toggle("Enable Grok", value=st.session_state.get("use_ai", False), key="grok_toggle")
            st.session_state.use_ai = st.session_state.grok_toggle
            api_key = st.text_input("API Key", type="password", value=st.session_state.get("grok_api_key", ""), key="grok_key")
            if api_key:
                st.session_state.grok_api_key = api_key
            st.caption("Get key from console.x.ai")
    
    # ── Data Management ──
    with st.sidebar.expander("Data", expanded=False):
        max_pages = st.slider("Pages to scrape", min_value=5, max_value=150, value=80, step=5, key="max_pages")
        min_content = st.slider("Min content length", min_value=20, max_value=200, value=30, step=10, key="min_content")
        if st.button("🔄 Scrape lict.edu.np", use_container_width=True):
            with st.spinner("Scraping LICT website..."):
                pages = scrape_site(max_pages=max_pages, min_content_length=min_content)
                save_pages(pages, DATA_PATH)
                upsert_scraped_pages(pages)
            st.success(f"✅ Saved {len(pages)} pages!")
            st.rerun()
        if st.button("🔄 Sync JSON to DB", use_container_width=True):
            count = sync_json_to_db()
            st.success(f"✅ Synced {count} pages!")
            st.rerun()
    
    # ── Bottom: User info + Knowledge base ──
    pages = load_knowledge_base()
    st.sidebar.divider()
    st.sidebar.metric("📄 Knowledge Base", f"{len(pages)} pages")
    if Path(DATA_PATH).exists():
        size_kb = Path(DATA_PATH).stat().st_size / 1024
        st.sidebar.caption(f"Data file: {size_kb:.0f} KB")
    
    # User info at the very bottom
    st.sidebar.divider()
    col_u1, col_u2, col_u3 = st.sidebar.columns([1, 4, 1])
    with col_u1:
        st.markdown(
            f"""
            <div style="
                width: 28px; height: 28px; border-radius: 50%;
                background: #444; display: flex; align-items: center; justify-content: center;
                color: #ececec; font-weight: 600; font-size: 0.8rem;
            ">{user['name'][0].upper()}</div>
            """,
            unsafe_allow_html=True,
        )
    with col_u2:
        st.markdown(f'<span style="font-size: 0.82rem; color: #b4b4b4;">{user["name"]}</span>', unsafe_allow_html=True)
    with col_u3:
        if st.button("🚪", key="signout_btn", help="Sign out"):
            st.session_state.clear()
            st.rerun()
    
    return pages


# ── Chat UI ──────────────────────────────────────────────────────────────

def render_chat(pages: list[dict]) -> None:
    model_type = st.session_state.get("model_type", "ollama")
    use_ai = st.session_state.get("use_ai", True)
    summarize_mode = st.session_state.get("summarize_mode", True)
    
    if model_type == "ollama":
        model_name = st.session_state.get("ollama_model", OLLAMA_MODEL)
        mode = f"Ollama `{model_name}`" if use_ai else "Local Retrieval"
    elif model_type == "gemini":
        mode = "Google Gemini" if use_ai else "Local Retrieval"
    elif model_type == "grok":
        mode = "Grok (xAI)" if use_ai else "Local Retrieval"
    else:
        mode = "Local Retrieval"
    
    st.markdown(
        f"""
        <div style="text-align: center; margin-bottom: 1rem;">
            <h1 style="margin: 0; font-size: 1.6rem; font-weight: 600;">🎓 LICT Assistant</h1>
            <p style="margin: 2px 0 0 0; color: #888; font-size: 0.85rem;">
                Mode: {mode} | Auto-summarize: {"On" if summarize_mode else "Off"}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    if not pages:
        st.warning("📭 No website data found. Go to **Sidebar → Data → Scrape** to build the knowledge base.", icon="⚠️")
        return
    
    if "messages" not in st.session_state:
        session_id = get_or_create_current_session()
        if session_id:
            msgs = load_session_messages(session_id)
            st.session_state.messages = msgs if msgs else [
                {"role": "assistant", "content": "👋 Hello! I'm the LICT assistant. Ask me anything about courses, admissions, notices, faculty, or any other information from the LICT website."}
            ]
        else:
            st.session_state.messages = [
                {"role": "assistant", "content": "👋 Hello! I'm the LICT assistant. Ask me anything about courses, admissions, notices, faculty, or any other information from the LICT website."}
            ]
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    prompt = st.chat_input("💬 Ask about LICT...", key="chat_input")
    
    if prompt:
        session_id = get_or_create_current_session()
        if not session_id:
            st.error("Could not create chat session. Please try signing out and back in.")
            return
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_message_to_db(session_id, "user", prompt)
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        model_type = st.session_state.get("model_type", "ollama")
        use_ai = st.session_state.get("use_ai", True)
        
        if model_type == "gemini":
            key = st.session_state.get("gemini_api_key", "")
            if key:
                import os
                os.environ["GEMINI_API_KEY"] = key
        elif model_type == "grok":
            key = st.session_state.get("grok_api_key", "")
            if key:
                import os
                os.environ["GROK_API_KEY"] = key
        
        with st.chat_message("assistant"):
            with st.spinner("💭 Thinking..."):
                answer = build_answer(
                    prompt,
                    pages,
                    st.session_state.messages,
                    use_ai=use_ai,
                    model_type=model_type,
                    model=st.session_state.get("ollama_model", OLLAMA_MODEL),
                )
            st.markdown(answer)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
        save_message_to_db(session_id, "assistant", answer)
        st.rerun()


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    st.markdown(CHATGPT_CSS, unsafe_allow_html=True)
    init_db()
    
    if "user" not in st.session_state:
        render_auth()
        return
    
    pages = render_sidebar()
    render_chat(pages)


if __name__ == "__main__":
    main()