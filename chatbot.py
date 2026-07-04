import os
import re
import logging
from collections import Counter
from typing import Optional

import requests
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

GREETINGS = {"hi", "hello", "hey", "namaste", "good morning", "good afternoon", "good evening", "namaskar"}

# Summary-related keywords that trigger summarization
SUMMARY_KEYWORDS = frozenset({
    "summarize", "summary", "summarise", "brief", "overview", "key points",
    "main points", "tl;dr", "tl;dr", "in short", "in brief", "shorten",
    "condense", "give me the gist", "what's important", "highlights",
    "main ideas", "key takeaways", "essence",
})

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

# Common stop words for query analysis
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "i", "in", "is", "it", "of", "on", "or", "please", "tell",
    "the", "to", "was", "were", "with", "you", "your", "we", "our",
    "its", "what", "which", "who", "whom", "this", "that", "these",
    "those", "am", "been", "being", "has", "have", "had", "do", "does",
    "did", "but", "if", "so", "than", "too", "very", "just", "about",
    "above", "after", "again", "all", "also", "any", "because", "can",
    "could", "each", "every", "few", "get", "got", "into", "like",
    "more", "most", "much", "must", "need", "no", "not", "now",
    "only", "other", "out", "over", "really", "same", "should",
    "some", "such", "than", "then", "there", "they", "thing",
    "things", "through", "under", "up", "use", "want", "way",
    "well", "where", "while", "why", "will", "would",
}

# Expanded stop words kept from original
STOP_WORDS_ORIGINAL = {
    "a", "an", "and", "are", "as", "at", "be", "for", "from", "how",
    "i", "in", "is", "it", "of", "on", "or", "please", "tell",
    "the", "to", "what", "with", "you",
}

# Core LICT-related terms that are important for matching
LICT_SPECIFIC_TERMS = frozenset({
    "lict", "lumbini", "ict", "campus", "college", "csit", "bca", "bim", "bhm",
    "tribhuvan", "university", "tu", "admission", "entrance", "exam", "result",
    "notice", "notice", "scholarship", "fee", "fees", "course", "program",
    "bachelor", "semester", "credit", "grade", "faculty", "staff", "teacher",
    "principal", "chairman", "contact", "phone", "email", "address",
    "location", "ganeshpur", "gaindakot", "nawalpur", "nawalparasi",
    "computer", "science", "information", "technology", "management",
    "hotel", "hospitality", "application", "syllabus", "curriculum",
    "internship", "project", "career", "job", "placement",
})


def is_greeting(message: str) -> bool:
    normalized = re.sub(r"[^a-z\s]", "", message.lower()).strip()
    return normalized in GREETINGS or normalized.split(" ")[0] in GREETINGS


def tokenize(text: str) -> list[str]:
    """Extract meaningful tokens from text."""
    return [
        word
        for word in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if len(word) > 1 and word not in STOP_WORDS
    ]


def extract_important_terms(text: str) -> list[str]:
    """
    Extract important terms including n-grams and key phrases.
    This helps match content even when wording differs.
    """
    text_lower = text.lower()
    tokens = re.findall(r"[a-zA-Z0-9]+", text_lower)
    
    important_terms = set()
    
    # Single words (excluding stop words)
    for word in tokens:
        if word not in STOP_WORDS and len(word) > 2:
            important_terms.add(word)
            # Add word stems (first 4-5 chars) for fuzzy matching
            if len(word) > 5:
                important_terms.add(word[:5])
                important_terms.add(word[:4])
    
    # Bigrams (important for course names like "computer science")
    for i in range(len(tokens) - 1):
        bigram = f"{tokens[i]} {tokens[i+1]}"
        important_terms.add(bigram)
    
    # Trigrams
    for i in range(len(tokens) - 2):
        trigram = f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"
        important_terms.add(trigram)
    
    return list(important_terms)


def generate_text_shingles(text: str, k: int = 3) -> set[str]:
    """
    Generate character-level k-shingles for fuzzy text matching.
    This allows matching even when word order or exact wording differs.
    """
    text_clean = re.sub(r"\s+", " ", text.lower()).strip()
    shingles = set()
    for i in range(len(text_clean) - k + 1):
        shingles.add(text_clean[i:i+k])
    return shingles


def compute_jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def score_page_fuzzy(question: str, page: dict) -> float:
    """
    Score a page's relevance to the question using multiple techniques:
    1. Exact keyword matching (weighted)
    2. N-gram (phrase) matching
    3. Character shingle similarity (for fuzzy/wording-different matching)
    4. LICT-specific term boosting
    """
    title = page.get("title", "").lower()
    content = page.get("content", "").lower()
    haystack = f"{title} {title} {content}"  # Title doubled for bonus
    
    question_lower = question.lower()
    question_terms = tokenize(question)
    
    if not question_terms:
        return 0.0
    
    score = 0.0
    
    # 1. Exact term matching with term frequency
    for term in question_terms:
        count = haystack.count(term)
        if count > 0:
            # Base score per occurrence
            term_score = count * 2.0
            # Title bonus
            if term in title:
                term_score *= 3.0
            # Boost LICT-specific terms
            if term in LICT_SPECIFIC_TERMS:
                term_score *= 1.5
            score += term_score
    
    # 2. N-gram matching (phrases like "computer science", "entrance exam")
    words = question_lower.split()
    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i+1]}"
        if bigram in haystack:
            score += 5.0
            if bigram in title:
                score += 10.0
    
    for i in range(len(words) - 2):
        trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
        if trigram in haystack:
            score += 8.0
            if trigram in title:
                score += 15.0
    
    # 3. Character shingle similarity for fuzzy matching
    # This catches cases where the wording is different but characters overlap
    q_shingles = generate_text_shingles(question, k=3)
    c_shingles = generate_text_shingles(content[:2000], k=3)  # First 2000 chars
    shingle_sim = compute_jaccard_similarity(q_shingles, c_shingles)
    score += shingle_sim * 15.0  # Weighted shingle contribution
    
    return score


def find_matching_pages(
    question: str,
    pages: list[dict],
    limit: int = 6,
    min_score: float = 0.5,
) -> list[dict]:
    """
    Find relevant pages using fuzzy and n-gram matching.
    Returns the most relevant pages sorted by relevance score.
    Always returns at least the top pages if ANY have a positive score.
    """
    if not pages:
        return []
    
    # Score all pages
    scored_pages = [
        (score_page_fuzzy(question, page), page)
        for page in pages
    ]
    
    # Sort by score descending
    scored_pages.sort(key=lambda x: x[0], reverse=True)
    
    # Filter pages with meaningful score
    top_score = scored_pages[0][0] if scored_pages else 0
    
    # Dynamic threshold: if top score is high, be more selective
    if top_score > 50:
        threshold = max(min_score, top_score * 0.15)
    else:
        threshold = min_score
    
    matches = [page for score, page in scored_pages if score > threshold]
    
    # If we have strong matches, return them limited
    if len(matches) >= 2:
        return matches[:limit]
    
    # If we have some matches, return what we have
    if matches:
        # Add more borderline pages up to limit
        borderline = [
            page for score, page in scored_pages[len(matches):limit]
            if score > 0
        ]
        matches.extend(borderline)
        return matches[:limit]
    
    # Last resort: check if any page has even partial character overlap
    question_terms = set(re.findall(r"[a-zA-Z]{4,}", question.lower()))
    for score, page in scored_pages[:limit]:
        content_lower = page.get("content", "").lower()
        content_terms = set(re.findall(r"[a-zA-Z]{4,}", content_lower))
        if question_terms & content_terms:
            matches.append(page)
    
    return matches[:limit]


def deduplicate_pages(pages: list[dict]) -> list[dict]:
    """
    Remove near-duplicate pages with very similar content.
    Keeps the version with the longest content.
    """
    if not pages:
        return []
    
    unique_pages = []
    seen_urls = set()
    seen_content_fingerprints = set()
    
    for page in pages:
        url = page.get("url", "")
        content = page.get("content", "")
        
        # Skip same exact URL
        if url in seen_urls:
            continue
        seen_urls.add(url)
        
        # Content fingerprint (first 100 chars normalized)
        fingerprint = re.sub(r"\s+", " ", content[:100].lower()).strip()
        if fingerprint in seen_content_fingerprints:
            continue
        seen_content_fingerprints.add(fingerprint)
        
        unique_pages.append(page)
    
    return unique_pages


def smart_truncate_context(pages: list[dict], max_total_chars: int = 6000) -> str:
    """
    Format context intelligently, allocating more characters to higher-ranked pages.
    """
    if not pages:
        return "No LICT website data available."
    
    # Calculate character budgets
    num_pages = len(pages)
    if num_pages == 1:
        budgets = [min(max_total_chars, 4000)]
    else:
        # First page gets more, rest share remaining
        first_budget = min(int(max_total_chars * 0.35), 3000)
        remaining = max_total_chars - first_budget
        other_budget = remaining // (num_pages - 1)
        other_budget = min(other_budget, 2000)
        budgets = [first_budget] + [other_budget] * (num_pages - 1)
    
    context_blocks = []
    for index, (page, budget) in enumerate(zip(pages, budgets), start=1):
        content = page.get("content", "")
        
        # If content is short, use it all
        if len(content) <= budget:
            truncated = content
        else:
            # Smart truncation at sentence boundary
            truncated = content[:budget]
            # Try to break at a sentence end
            last_period = max(
                truncated.rfind(". "),
                truncated.rfind("? "),
                truncated.rfind("! "),
                truncated.rfind("\n"),
            )
            if last_period > budget * 0.5:  # Only if we have enough content
                truncated = truncated[:last_period + 1]
            else:
                # Break at word boundary
                truncated = truncated.rsplit(" ", 1)[0] + "..."
        
        context_blocks.append(
            f"[Source {index}]\n"
            f"Title: {page.get('title', 'Untitled')}\n"
            f"Date: {page.get('date') or 'No date found'}\n"
            f"URL: {page.get('url', '')}\n"
            f"Content: {truncated}"
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
    context = smart_truncate_context(context_pages)
    system_prompt = (
        "You are an intelligent assistant for Lumbini ICT Campus (LICT). "
        "You have access to the complete LICT website knowledge base. "
        "Answer questions using the provided LICT website context comprehensively. "
        "Search through ALL provided pages for relevant information. "
        "Be conversational, accurate, and concise. "
        "If you find relevant information in the provided context, use it. "
        "Only say 'the scraped LICT data does not include it' if you've checked all context and found nothing. "
        "Do not invent admissions dates, fees, notices, phone numbers, or policies - only use what's in the context. "
        "When useful, reference the source page number or title. "
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
    
    context = smart_truncate_context(context_pages)
    system_prompt = (
        "You are an intelligent assistant for Lumbini ICT Campus (LICT). "
        "You have access to the complete LICT website knowledge base. "
        "Answer questions using the provided LICT website context comprehensively. "
        "Search through ALL provided pages for relevant information. "
        "Be conversational, accurate, and concise. "
        "If you find relevant information in the provided context, use it. "
        "Only say 'the scraped LICT data does not include it' if you've checked all context and found nothing. "
        "Do not invent admissions dates, fees, notices, phone numbers, or policies - only use what's in the context. "
        "When useful, reference the source page number or title. "
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
    
    context = smart_truncate_context(context_pages)
    system_prompt = (
        "You are an intelligent assistant for Lumbini ICT Campus (LICT). "
        "You have access to the complete LICT website knowledge base. "
        "Answer questions using the provided LICT website context comprehensively. "
        "Search through ALL provided pages for relevant information. "
        "Be conversational, accurate, and concise. "
        "If you find relevant information in the provided context, use it. "
        "Only say 'the scraped LICT data does not include it' if you've checked all context and found nothing. "
        "Do not invent admissions dates, fees, notices, phone numbers, or policies - only use what's in the context. "
        "When useful, reference the source page number or title. "
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


def is_summary_request(message: str) -> bool:
    """Check if the user is asking for a summary of the content."""
    normalized = re.sub(r"[^a-z\s]", "", message.lower()).strip()
    words = set(normalized.split())
    
    # Check for explicit summary keywords
    if words & SUMMARY_KEYWORDS:
        return True
    
    # Check for combination patterns like "summarize X", "give me summary of Y"
    summary_patterns = [
        r"\bsummarize\s+",
        r"\bsummary\s+of\s+",
        r"\bsummarise\s+",
        r"\bkey\s+points\s+of\s+",
        r"\boverview\s+of\s+",
        r"\bmain\s+points\s+of\s+",
        r"\bhighlights?\s+of\s+",
        r"\bwhat'?s\s+important\s+about\s+",
    ]
    for pattern in summary_patterns:
        if re.search(pattern, normalized):
            return True
    
    return False


def build_summary_answer(matches: list[dict]) -> str:
    """Build a summarized answer from matched pages."""
    if not matches:
        return "No content available to summarize."
    
    # Combine content from all matched pages
    combined_content = "\n\n".join(
        page.get("content", "")[:2000] for page in matches
    )
    
    # Create bullet point summary
    lines = ["**Summary of relevant LICT information:**\n"]
    
    for i, page in enumerate(matches, 1):
        title = page.get("title", "Untitled")
        content = page.get("content", "")
        
        # Extract key sentences (first few sentences)
        sentences = re.split(r"[.?!]+", content)
        key_sentences = []
        seen_keywords = set()
        
        for sentence in sentences[:15]:  # Check first 15 sentences
            sentence = sentence.strip()
            if not sentence or len(sentence) < 30:
                continue
            
            # Simple keyword check for relevance
            sentence_lower = sentence.lower()
            key_terms = any(
                term in sentence_lower
                for term in ["course", "program", "duration", "eligibility", "credit",
                           "semester", "admission", "fee", "syllabus", "curriculum",
                           "technology", "management", "science", "hotel", "application",
                           "computer", "information"]
            )
            
            if key_terms:
                # Avoid duplicate content
                sentence_tokens = set(sentence.lower().split())
                if not sentence_tokens & seen_keywords:
                    key_sentences.append(sentence)
                    seen_keywords.update(sentence_tokens)
        
        if key_sentences:
            summary = " ".join(key_sentences[:5])  # Limit to 5 key sentences per page
            if len(summary) > 400:
                summary = summary[:400].rsplit(" ", 1)[0] + "..."
            lines.append(f"**{title}**")
            lines.append(f"{summary}\n")
    
    return "\n".join(lines)


def build_retrieval_answer(matches: list[dict]) -> str:
    lines = ["Here is what I found from the LICT website data:"]
    for page in matches[:5]:
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

def call_ollama_for_summary(
    matches: list[dict],
    model: str = OLLAMA_MODEL,
) -> str:
    """Call Ollama with a summarization system prompt."""
    context = smart_truncate_context(matches)
    system_prompt = (
        "You are an intelligent assistant for Lumbini ICT Campus (LICT). "
        "You have access to the complete LICT website knowledge base. "
        "Your task is to provide a CLEAR, CONCISE SUMMARY of the provided content. "
        "Extract the key information and present it in an organized, easy-to-read format. "
        "Use bullet points or numbered lists where appropriate. "
        "Focus on the most important facts, dates, requirements, and details. "
        "Include the source page titles as references. "
        "Be accurate - only use information from the context."
    )
    user_prompt = (
        f"Please summarize the following LICT website content:\n\n{context}\n\n"
        f"Provide a concise summary highlighting the key information."
    )
    messages = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": user_prompt})

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
            },
        },
        timeout=90,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("message", {}).get("content", "").strip()


def call_gemini_for_summary(matches: list[dict]) -> str:
    """Call Gemini with a summarization system prompt."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set in .env file")
    
    import google.generativeai as genai
    
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-pro")
    
    context = smart_truncate_context(matches)
    system_prompt = (
        "You are an intelligent assistant for Lumbini ICT Campus (LICT). "
        "Your task is to provide a CLEAR, CONCISE SUMMARY of the provided content. "
        "Extract the key information and present it in an organized, easy-to-read format. "
        "Use bullet points or numbered lists where appropriate. "
        "Focus on the most important facts, dates, requirements, and details. "
        "Be accurate - only use information from the context."
    )
    user_prompt = (
        f"{system_prompt}\n\n{context}\n\n"
        f"Please summarize this LICT website content with key points."
    )
    
    response = model.generate_content(user_prompt)
    return response.text.strip()


def call_grok_for_summary(matches: list[dict]) -> str:
    """Call Grok with a summarization system prompt."""
    if not GROK_API_KEY:
        raise ValueError("GROK_API_KEY not set in .env file")
    
    context = smart_truncate_context(matches)
    system_prompt = (
        "You are an intelligent assistant for Lumbini ICT Campus (LICT). "
        "Your task is to provide a CLEAR, CONCISE SUMMARY of the provided content. "
        "Extract the key information and present it in an organized, easy-to-read format. "
        "Use bullet points or numbered lists where appropriate. "
        "Focus on the most important facts, dates, requirements, and details."
    )
    user_prompt = (
        f"Please summarize the following LICT website content:\n\n{context}\n\n"
        f"Provide a concise summary with key points."
    )
    
    messages = [{"role": "system", "content": system_prompt}]
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
            "temperature": 0.1,
        },
        timeout=90,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()


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

    # Find relevant pages using improved fuzzy matching
    raw_matches = find_matching_pages(question, pages)
    
    # Deduplicate before sending to AI
    matches = deduplicate_pages(raw_matches)

    if not matches or len(pages) == 0:
        return (
            "I could not find relevant information in the LICT database. "
            "Try asking with different keywords or request the data to be refreshed from the sidebar. "
            "Available topics: notices, courses, admissions, contact information, and campus details."
        )

    # Check if user wants a summary
    if is_summary_request(question):
        if not use_ai:
            return build_summary_answer(matches)
        
        try:
            if model_type.lower() == "gemini":
                answer = call_gemini_for_summary(matches)
            elif model_type.lower() == "grok":
                answer = call_grok_for_summary(matches)
            else:  # ollama
                answer = call_ollama_for_summary(matches, model=model)
            
            if answer:
                return f"**📋 Summary of LICT Information:**\n\n{answer}"
        except requests.RequestException as exc:
            return (
                f"Error reaching {model_type.upper()} for summarization.\n\n"
                f"{build_summary_answer(matches)}"
            )
        except (ImportError, ValueError) as exc:
            return (
                f"Error: {str(exc)}\n\n"
                f"{build_summary_answer(matches)}"
            )
        
        return build_summary_answer(matches)

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


# Utility for testing
def test_query(query: str, pages: list[dict] | None = None) -> None:
    """Test how a query would be matched against pages."""
    if pages is None:
        from scraper import load_pages
        pages = load_pages()
    
    print(f"\nQuery: '{query}'")
    print(f"Tokens: {tokenize(query)}")
    print(f"Important terms: {extract_important_terms(query)[:15]}")
    
    matches = find_matching_pages(query, pages, limit=5)
    print(f"\nTop {len(matches)} matching pages:")
    for i, page in enumerate(matches, 1):
        score = score_page_fuzzy(query, page)
        content_preview = page.get("content", "")[:100].replace("\n", " ")
        print(f"{i}. Score={score:.1f} | {page.get('title', 'N/A')}")
        print(f"   URL: {page.get('url', '')}")
        print(f"   Preview: {content_preview}...")


if __name__ == "__main__":
    # Test mode
    from scraper import load_pages
    all_pages = load_pages()
    print(f"Loaded {len(all_pages)} pages from data file")
    
    test_queries = [
        "What courses are offered at LICT?",
        "Tell me about BSc CSIT",
        "What is the admission process?",
        "How do I contact the college?",
        "Show me notice and news",
        "What is the fee for BCA?",
        "Who is the principal?",
        "Tell me about scholarship programs",
    ]
    
    for query in test_queries:
        test_query(query, all_pages)
        print("\n" + "="*60)