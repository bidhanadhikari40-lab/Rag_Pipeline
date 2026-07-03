import os
import re
from collections import Counter

import requests
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

GREETINGS = {"hi", "hello", "hey", "namaste", "good morning", "good afternoon", "good evening"}

# Ollama Configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")

# Gemini Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Grok Configuration
GROK_API_KEY = os.getenv("GROK_API_KEY", "")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-beta")

# Default model type
DEFAULT_MODEL_TYPE = os.getenv("AI_MODEL_TYPE", "ollama")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "please",
    "tell",
    "the",
    "to",
    "what",
    "with",
    "you",
}


def is_greeting(message: str) -> bool:
    normalized = re.sub(r"[^a-z\s]", "", message.lower()).strip()
    return normalized in GREETINGS or normalized.split(" ")[0] in GREETINGS


def tokenize(text: str) -> list[str]:
    return [
        word
        for word in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if len(word) > 2 and word not in STOP_WORDS
    ]


def score_page(query_terms: Counter, page: dict) -> int:
    haystack = f"{page.get('title', '')} {page.get('content', '')}".lower()
    score = 0
    for term, weight in query_terms.items():
        if term in haystack:
            title_bonus = 4 if term in page.get("title", "").lower() else 0
            score += weight + title_bonus + haystack.count(term)
    return score


def find_matching_pages(question: str, pages: list[dict], limit: int = 4) -> list[dict]:
    """
    Find matching pages using keyword matching.
    Falls back to returning all pages if no matches found (better than nothing).
    """
    query_terms = Counter(tokenize(question))
    if not query_terms:
        # If no keywords extracted, return all pages
        return pages[:limit * 2]

    ranked = sorted(
        ((score_page(query_terms, page), page) for page in pages),
        key=lambda item: item[0],
        reverse=True,
    )
    
    # Filter pages with score > 0
    matches = [page for score, page in ranked if score > 0]
    
    # If we found matches, return top ones
    if matches:
        return matches[:limit]
    
    # Fallback: If no keyword matches found, return all pages
    # This ensures the AI has access to everything to answer the question
    return pages


def format_context(pages: list[dict], max_chars_per_page: int = 1200) -> str:
    context_blocks = []
    for index, page in enumerate(pages, start=1):
        content = page.get("content", "")[:max_chars_per_page]
        context_blocks.append(
            f"[Source {index}]\n"
            f"Title: {page.get('title', 'Untitled')}\n"
            f"Date: {page.get('date') or 'No date found'}\n"
            f"URL: {page.get('url', '')}\n"
            f"Content: {content}"
        )
    return "\n\n".join(context_blocks)


def format_history(history: list[dict] | None, limit: int = 6) -> list[dict]:
    if not history:
        return []

    formatted = []
    for message in history[-limit:]:
        role = message.get("role")
        content = message.get("content", "")
        if role in {"user", "assistant"} and content:
            formatted.append({"role": role, "content": content})
    return formatted


def call_ollama(
    question: str,
    context_pages: list[dict],
    history: list[dict] | None = None,
    model: str = OLLAMA_MODEL,
) -> str:
    context = format_context(context_pages)
    system_prompt = (
        "You are an intelligent assistant for Lumbini ICT Campus (LICT). "
        "You have access to the complete LICT website knowledge base. "
        "Answer questions using the provided LICT website context comprehensively. "
        "If the user asks about something, search through ALL the provided pages for relevant information. "
        "Be conversational, accurate, and concise. "
        "If you find relevant information in the provided context, use it. "
        "Only say 'the scraped LICT data does not include it' if you've checked all context and found nothing. "
        "Do not invent admissions dates, fees, notices, phone numbers, or policies - only use what's in the context. "
        "When useful, reference the source page or topic. "
        "Combine information from multiple pages if needed to give a complete answer."
    )
    user_prompt = (
        f"LICT Website Knowledge Base:\n{context}\n\n"
        f"User Question: {question}\n\n"
        f"Please answer based on the above knowledge base."
    )
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(format_history(history))
    messages.append({"role": "user", "content": user_prompt})

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
            },
        },
        timeout=90,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("message", {}).get("content", "").strip()


def call_gemini(
    question: str,
    context_pages: list[dict],
    history: list[dict] | None = None,
) -> str:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set in .env file")
    
    import google.generativeai as genai
    
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-pro")
    
    context = format_context(context_pages)
    system_prompt = (
        "You are an intelligent assistant for Lumbini ICT Campus (LICT). "
        "You have access to the complete LICT website knowledge base. "
        "Answer questions using the provided LICT website context comprehensively. "
        "If the user asks about something, search through ALL the provided pages for relevant information. "
        "Be conversational, accurate, and concise. "
        "If you find relevant information in the provided context, use it. "
        "Only say 'the scraped LICT data does not include it' if you've checked all context and found nothing. "
        "Do not invent admissions dates, fees, notices, phone numbers, or policies - only use what's in the context. "
        "When useful, reference the source page or topic. "
        "Combine information from multiple pages if needed to give a complete answer."
    )
    user_prompt = (
        f"{system_prompt}\n\n"
        f"LICT Website Knowledge Base:\n{context}\n\n"
        f"User Question: {question}\n\n"
        f"Please answer based on the above knowledge base."
    )
    
    response = model.generate_content(user_prompt)
    return response.text.strip()


def call_grok(
    question: str,
    context_pages: list[dict],
    history: list[dict] | None = None,
) -> str:
    if not GROK_API_KEY:
        raise ValueError("GROK_API_KEY not set in .env file")
    
    context = format_context(context_pages)
    system_prompt = (
        "You are an intelligent assistant for Lumbini ICT Campus (LICT). "
        "You have access to the complete LICT website knowledge base. "
        "Answer questions using the provided LICT website context comprehensively. "
        "If the user asks about something, search through ALL the provided pages for relevant information. "
        "Be conversational, accurate, and concise. "
        "If you find relevant information in the provided context, use it. "
        "Only say 'the scraped LICT data does not include it' if you've checked all context and found nothing. "
        "Do not invent admissions dates, fees, notices, phone numbers, or policies - only use what's in the context. "
        "When useful, reference the source page or topic. "
        "Combine information from multiple pages if needed to give a complete answer."
    )
    user_prompt = (
        f"LICT Website Knowledge Base:\n{context}\n\n"
        f"User Question: {question}\n\n"
        f"Please answer based on the above knowledge base."
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(format_history(history))
    messages.append({"role": "user", "content": user_prompt})
    
    response = requests.post(
        "https://api.x.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROK_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": GROK_MODEL,
            "messages": messages,
            "temperature": 0.2,
        },
        timeout=90,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()


def build_retrieval_answer(matches: list[dict]) -> str:
    lines = ["Here is what I found from the LICT website data:"]
    for page in matches:
        content = page.get("content", "")
        snippet = content[:550].rsplit(" ", 1)[0]
        if len(content) > len(snippet):
            snippet += "..."
        date = page.get("date") or "No date found"
        lines.append(
            f"\n**{page.get('title', 'Untitled')}**\n"
            f"{snippet}\n"
            f"Date: {date}\n"
            f"Source: {page.get('url', '')}"
        )

    return "\n".join(lines)


def build_answer(
    question: str,
    pages: list[dict],
    history: list[dict] | None = None,
    use_ai: bool = True,
    model_type: str = "ollama",
    model: str = OLLAMA_MODEL,
) -> str:
    if is_greeting(question):
        return "Hello! I can help you find LICT information about notices, courses, admissions, contacts, and general college details. Ask me anything!"

    if not tokenize(question):
        return "Please ask about LICT - courses, admissions, notices, contacts, campus info, or any other LICT-related topic."

    matches = find_matching_pages(question, pages)

    if not matches or len(pages) == 0:
        return (
            "I could not find relevant information in the LICT database. "
            "Try asking with different keywords or request the data to be refreshed from the sidebar. "
            "Available topics: notices, courses, admissions, contact information, and campus details."
        )

    if not use_ai:
        return build_retrieval_answer(matches)

    try:
        if model_type.lower() == "gemini":
            answer = call_gemini(question, matches, history)
        elif model_type.lower() == "grok":
            answer = call_grok(question, matches, history)
        else:  # ollama
            answer = call_ollama(question, matches, history, model=model)
        
        if answer:
            return answer
    except requests.RequestException as exc:
        model_name = model if model_type.lower() == "ollama" else model_type.upper()
        return (
            f"I found relevant LICT data, but could not reach {model_name}. "
            f"Make sure your API key is configured and the service is accessible.\n\n"
            f"Technical detail: `{exc}`\n\n"
            f"{build_retrieval_answer(matches)}"
        )
    except (ImportError, ValueError) as exc:
        return (
            f"Error: {str(exc)}\n\n"
            f"Please configure the API key in your Streamlit sidebar for {model_type.upper()}.\n\n"
            f"{build_retrieval_answer(matches)}"
        )

    return build_retrieval_answer(matches)
