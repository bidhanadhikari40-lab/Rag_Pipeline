# LICT Website Chatbot

This project scrapes `lict.edu.np`, stores cleaned website content as JSON and SQLite rows, and exposes a Streamlit chatbot behind a simple registered-user login system.

## Features

- BeautifulSoup-based static website scraper.
- Structured JSON export at `data/lict_pages.json`.
- SQLite database at `data/chatbot.db`.
- `admin_info` table with hashed passwords.
- Authenticated Streamlit UI with a ChatGPT-like dark theme.
- Conversation history stored in `st.session_state`.
- Rule-based greeting handling.
- Local keyword retrieval over scraped LICT website data.

## Run Locally

```powershell
pip install -r requirements.txt
streamlit run app.py
```

Open the app, create an account, sign in, then click **Scrape lict.edu.np** in the sidebar.

You can also run the scraper directly:

```powershell
python scraper.py
```

## Deployment

1. Push this folder to GitHub.
2. Go to Streamlit Community Cloud.
3. Create a new app and select `app.py` as the main file.
4. Keep secrets out of source code. Add API keys in Streamlit Cloud secrets only if you later connect an LLM provider.
