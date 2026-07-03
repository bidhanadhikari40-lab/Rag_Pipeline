from pathlib import Path

import streamlit as st

from chatbot import OLLAMA_MODEL, build_answer
from database import (
    authenticate,
    create_admin,
    init_db,
    load_pages_from_db,
    upsert_scraped_pages,
)
from scraper import DATA_PATH, load_pages, save_pages, scrape_site


st.set_page_config(page_title="LICT Chatbot", page_icon=":speech_balloon:", layout="wide")


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


def render_auth() -> None:
    st.title("LICT Chatbot")
    st.caption("Sign in to access the scraped website knowledge base.")

    login_tab, register_tab = st.tabs(["Login", "Register"])

    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", use_container_width=True)

        if submitted:
            user = authenticate(email, password)
            if user:
                st.session_state.user = dict(user)
                st.session_state.messages = []
                st.rerun()
            st.error("Invalid email or password.")

    with register_tab:
        with st.form("register_form"):
            name = st.text_input("Name")
            contact_detail = st.text_input("Contact detail")
            reg_email = st.text_input("Email", key="register_email")
            reg_password = st.text_input("Password", type="password", key="register_password")
            address = st.text_area("Address")
            registered = st.form_submit_button("Create account", use_container_width=True)

        if registered:
            ok, message = create_admin(name, contact_detail, reg_email, reg_password, address)
            if ok:
                st.success(message)
            else:
                st.error(message)


def render_sidebar() -> list[dict]:
    st.sidebar.title("LICT Chatbot")
    st.sidebar.write(f"Signed in as **{st.session_state.user['name']}**")

    if st.sidebar.button("Sign out", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("⚙️ AI Model Configuration")
    
    # Model type selection
    model_type = st.sidebar.selectbox(
        "Choose AI Model",
        options=["Ollama (Local)", "Gemini (Google)", "Grok (xAI)"],
        index=0,
        help="Select which AI model to use for answering questions",
    )
    
    # Map display names to actual model types
    model_type_map = {
        "Ollama (Local)": "ollama",
        "Gemini (Google)": "gemini",
        "Grok (xAI)": "grok",
    }
    st.session_state.model_type = model_type_map[model_type]
    
    # Model-specific configuration
    if st.session_state.model_type == "ollama":
        st.sidebar.info("ℹ️ Ollama runs locally on your machine")
        st.session_state.use_ai = st.sidebar.toggle(
            "Enable Ollama",
            value=st.session_state.get("use_ai", True),
            help="Use local Ollama model for intelligent answers",
        )
        st.session_state.ollama_model = st.sidebar.text_input(
            "Ollama Model Name",
            value=st.session_state.get("ollama_model", OLLAMA_MODEL),
            help="Install with: ollama pull gemma3:4b",
        )
        st.sidebar.caption("Make sure Ollama is running on http://localhost:11434")
        
    elif st.session_state.model_type == "gemini":
        st.sidebar.info("ℹ️ Requires Google Gemini API key")
        st.session_state.use_ai = st.sidebar.toggle(
            "Enable Gemini",
            value=st.session_state.get("use_ai", False),
            help="Use Google Gemini API for intelligent answers",
        )
        api_key = st.sidebar.text_input(
            "Gemini API Key",
            type="password",
            value=st.session_state.get("gemini_api_key", ""),
            help="Get your key from https://ai.google.dev/",
        )
        if api_key:
            st.session_state.gemini_api_key = api_key
        st.sidebar.caption("🔑 Keep your API key private")
        
    elif st.session_state.model_type == "grok":
        st.sidebar.info("ℹ️ Requires Grok API key from xAI")
        st.session_state.use_ai = st.sidebar.toggle(
            "Enable Grok",
            value=st.session_state.get("use_ai", False),
            help="Use Grok API for intelligent answers",
        )
        api_key = st.sidebar.text_input(
            "Grok API Key",
            type="password",
            value=st.session_state.get("grok_api_key", ""),
            help="Get your key from https://console.x.ai/",
        )
        if api_key:
            st.session_state.grok_api_key = api_key
        st.sidebar.caption("🔑 Keep your API key private")

    st.sidebar.divider()
    st.sidebar.subheader("📚 Website Data")

    max_pages = st.sidebar.slider("Pages to scrape", min_value=5, max_value=80, value=35, step=5)
    if st.sidebar.button("Scrape lict.edu.np", use_container_width=True):
        with st.spinner("Scraping LICT website..."):
            pages = scrape_site(max_pages=max_pages)
            save_pages(pages, DATA_PATH)
            upsert_scraped_pages(pages)
        st.sidebar.success(f"Saved {len(pages)} pages.")
        st.rerun()

    if st.sidebar.button("Sync JSON to database", use_container_width=True):
        count = sync_json_to_db()
        st.sidebar.success(f"Synced {count} pages.")
        st.rerun()

    pages = load_knowledge_base()
    st.sidebar.metric("Knowledge pages", len(pages))
    if Path(DATA_PATH).exists():
        st.sidebar.caption(f"JSON: {DATA_PATH}")

    return pages


def render_chat(pages: list[dict]) -> None:
    st.title("LICT Website Chatbot")
    
    model_type = st.session_state.get("model_type", "ollama")
    use_ai = st.session_state.get("use_ai", True)
    
    if model_type == "ollama":
        model_name = st.session_state.get("ollama_model", OLLAMA_MODEL)
        mode = f"Ollama `{model_name}`" if use_ai else "local keyword retrieval"
    elif model_type == "gemini":
        mode = "Google Gemini" if use_ai else "local keyword retrieval"
    elif model_type == "grok":
        mode = "Grok (xAI)" if use_ai else "local keyword retrieval"
    else:
        mode = "local keyword retrieval"
    
    st.caption(f"Ask about LICT notices, courses, admissions, contact details, and website pages. Mode: {mode}.")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! Ask me anything from the scraped LICT website data.",
            }
        ]

    if not pages:
        st.info("No website data found yet. Use **Scrape lict.edu.np** in the sidebar to build the knowledge base.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask about LICT...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = build_answer(
                    prompt,
                    pages,
                    st.session_state.messages,
                    use_ai=st.session_state.get("use_ai", True),
                    model_type=st.session_state.get("model_type", "ollama"),
                    model=st.session_state.get("ollama_model", OLLAMA_MODEL),
                )
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


def main() -> None:
    init_db()
    if "user" not in st.session_state:
        render_auth()
        return

    pages = render_sidebar()
    render_chat(pages)


if __name__ == "__main__":
    main()
