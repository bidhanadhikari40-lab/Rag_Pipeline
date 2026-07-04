# LICT Chatbot - Multi-Model AI Setup Guide

## Quick Start

The LICT Chatbot now supports three AI models:
1. **Ollama** (Local - No API key needed)
2. **Google Gemini** (Cloud - Requires API key)
3. **Grok** (xAI - Requires API key)

## Configuration

### Step 1: Copy and Configure .env File

A `.env.example` file is provided. Copy it to create your `.env` file:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys (if using Gemini or Grok).

### Step 2: Set Default AI Model

In `.env`, set your preferred model:

```env
AI_MODEL_TYPE=ollama  # Options: ollama, gemini, grok
```

---

## Model-Specific Setup

### Option 1: Ollama (Recommended for Local Use)

**Advantages:**
- ✅ No API key needed
- ✅ Completely private (runs locally)
- ✅ Fast responses

**Setup:**

1. Download and install Ollama from [ollama.ai](https://ollama.ai)

2. Pull the Gemma model:
```bash
ollama pull gemma3:4b
```

3. Start Ollama:
```bash
ollama serve
```

4. Configure in `.env`:
```env
AI_MODEL_TYPE=ollama
OLLAMA_URL=http://localhost:11434/api/chat
OLLAMA_MODEL=gemma3:4b
```

---

### Option 2: Google Gemini

**Advantages:**
- ✅ Powerful language model
- ✅ Better for complex queries
- ✅ Cloud-based (latest model versions)

**Setup:**

1. Get your free API key:
   - Go to [ai.google.dev](https://ai.google.dev)
   - Click "Get API key"
   - Create a new API key

2. Add to `.env`:
```env
AI_MODEL_TYPE=gemini
GEMINI_API_KEY=your_api_key_here
```

3. Requirements:
   - `pip install google-generativeai` (already included)

---

### Option 3: Grok (xAI)

**Advantages:**
- ✅ Newest AI model
- ✅ Real-time information access
- ✅ Very capable reasoning

**Setup:**

1. Get your API key:
   - Go to [console.x.ai](https://console.x.ai)
   - Create API credentials
   - Copy your API key

2. Add to `.env`:
```env
AI_MODEL_TYPE=grok
GROK_API_KEY=your_api_key_here
GROK_MODEL=grok-beta
```

---

## Running the Chatbot

### Option 1: Via Terminal (Bypass Device Guard)

```powershell
python -m streamlit run webscrap.py
```

### Option 2: Via Batch File

```bash
run_chatbot.bat
```

### Option 3: Via PowerShell Script

```powershell
.\run_chatbot.ps1
```

The app will open at: **http://localhost:8501**

---

## Switching Models in the App

1. Log in to the chatbot
2. Look at the left sidebar under "⚙️ AI Model Configuration"
3. Select your preferred model from the dropdown
4. Enable it with the toggle
5. Provide API key if required (for Gemini/Grok)

---

## Troubleshooting

### "GEMINI_API_KEY not set"
- Make sure your `.env` file has the correct Gemini API key
- Restart the Streamlit app after editing `.env`

### "GROK_API_KEY not set"
- Verify your Grok API key in `.env`
- Check that the API key has proper permissions

### Ollama not connecting
- Make sure Ollama is running: `ollama serve`
- Verify the model is installed: `ollama list`
- Check OLLAMA_URL in `.env` matches your setup

### API Rate Limits
- Gemini: 60 requests per minute (free tier)
- Grok: Check your plan limits
- Ollama: Unlimited (local)

---

## File Structure

```
Web Scraping/
├── .env                    # Your configuration (API keys)
├── .env.example           # Example configuration template
├── requirements.txt       # Python dependencies
├── chatbot.py            # AI logic (Ollama/Gemini/Grok)
├── webscrap.py           # Streamlit UI
├── database.py           # User authentication
├── scraper.py            # Website scraper
├── run_chatbot.bat       # Batch launcher
├── run_chatbot.ps1       # PowerShell launcher
└── data/
    ├── lict_pages.json   # Scraped website data
    └── chatbot.db        # User database
```

---

## Security Notes

🔒 **Never commit your `.env` file to version control!**

- `.env` is listed in `.gitignore` by default
- Keep API keys private and secure
- Rotate keys periodically
- Don't share `.env` files

---

## Support

For issues:
1. Check that your API keys are correct
2. Ensure required services are running (Ollama for local mode)
3. Check internet connection for cloud models
4. Review error messages in the Streamlit console

