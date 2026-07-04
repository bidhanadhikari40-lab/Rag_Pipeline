import json
import re
from collections import deque
from datetime import date
from pathlib import Path
from time import sleep
from urllib.parse import urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://lict.edu.np/"
DATA_PATH = Path("data/lict_pages.json")
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; LICTChatbotScraper/2.0; "
        "+https://lict.edu.np/)"
    )
}

# Low-value URL patterns to skip (tag pages, category pages, etc.)
SKIP_URL_PATTERNS = [
    r"/tag/",
    r"/category/",
    r"/page/",
    r"\?page=",
    r"#",
    r"/wp-",
    r"/feed",
    r"/author/",
]

# High-priority paths that should always be scraped
HIGH_PRIORITY_PATHS = [
    "/contact",
    "/contactus",
    "/contact-us",
    "/notice",
    "/notices",
    "/notice-board",
    "/admission",
    "/admissions",
    "/faculty",
    "/staff",
    "/aboutus",
    "/about-us",
    "/about",
    "/download",
    "/downloads",
    "/scholarship",
    "/scholarships",
    "/gallery",
    "/blog",
    "/news",
    "/event",
    "/events",
    "/research",
    "/career",
    "/careers",
    "/testimonial",
    "/testimonials",
]


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_same_site(url: str, base_url: str = BASE_URL) -> bool:
    base_host = urlparse(base_url).netloc.lower().removeprefix("www.")
    host = urlparse(url).netloc.lower().removeprefix("www.")
    return host == base_host


def normalize_url(url: str, base_url: str = BASE_URL) -> str:
    absolute_url = urljoin(base_url, url)
    absolute_url, _fragment = urldefrag(absolute_url)
    parsed = urlparse(absolute_url)
    return parsed._replace(query="").geturl().rstrip("/")


def should_skip_url(url: str) -> bool:
    """Check if a URL matches low-value patterns."""
    path = urlparse(url).path.lower()
    for pattern in SKIP_URL_PATTERNS:
        if re.search(pattern, path):
            return True
    return False


def is_high_priority(url: str) -> bool:
    """Check if a URL is a high-priority page."""
    path = urlparse(url).path.lower().rstrip("/")
    for priority_path in HIGH_PRIORITY_PATHS:
        if path == priority_path or path.rstrip("/") == priority_path:
            return True
    # Also check if path starts with a high-priority prefix
    for priority_path in HIGH_PRIORITY_PATHS:
        if path.startswith(priority_path + "/"):
            return True
    return False


def extract_date(text: str) -> str:
    patterns = [
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b",
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b",
        r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(0)
    return ""


def page_content_length(page: dict) -> int:
    """Return the length of meaningful content to detect duplicates."""
    return len(page.get("content", ""))


def is_duplicate_page(page: dict, existing_pages: list[dict], threshold: float = 0.85) -> bool:
    """
    Check if a page is substantially similar to an existing page.
    Uses content overlap ratio.
    """
    content_a = page.get("content", "").lower()
    title_a = page.get("title", "").lower()
    
    for existing in existing_pages:
        content_b = existing.get("content", "").lower()
        title_b = existing.get("title", "").lower()
        
        # Check title similarity
        if title_a and title_b and title_a == title_b:
            # Same title - check content overlap
            words_a = set(content_a.split())
            words_b = set(content_b.split())
            if not words_a or not words_b:
                continue
            overlap = len(words_a & words_b) / max(len(words_a), len(words_b))
            if overlap > threshold:
                return True
    
    return False


def parse_page(html: str, url: str) -> tuple[dict, list[str]]:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "iframe", "form"]):
        tag.decompose()

    title_tag = soup.find(["h1", "h2"]) or soup.find("title")
    title = clean_text(title_tag.get_text(" ")) if title_tag else url

    content_root = soup.find("main") or soup.find("article") or soup.body or soup
    
    # Expanded set of content-bearing tags
    text_parts = []
    for tag in content_root.find_all([
        "h1", "h2", "h3", "h4", "h5", "h6",
        "p", "li", "td", "th", "div", "span", "section",
        "blockquote", "pre", "code", "strong", "em",
        "dt", "dd", "caption", "address"
    ]):
        value = clean_text(tag.get_text(" "))
        # Skip very short snippets and navigation/boilerplate
        if value and len(value) > 2:
            text_parts.append(value)

    content = clean_text(" ".join(dict.fromkeys(text_parts)))
    
    page = {
        "title": title or "Untitled",
        "content": content,
        "date": extract_date(content) or str(date.today()),
        "url": url,
    }

    links = []
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        if href.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue
        normalized = normalize_url(href, url)
        if is_same_site(normalized) and not re.search(
            r"\.(jpg|jpeg|png|gif|pdf|zip|rar|doc|docx|xls|xlsx|ppt|pptx)$",
            normalized, re.I
        ):
            links.append(normalized)

    return page, links


def scrape_site(
    start_url: str = BASE_URL,
    max_pages: int = 100,
    delay_seconds: float = 0.4,
    min_content_length: int = 30,
) -> list[dict]:
    start_url = normalize_url(start_url)
    
    # Use two queues: high priority first, then regular
    high_priority_queue = deque()
    regular_queue = deque([start_url])
    
    visited = set()
    pages = []
    urls_seen = set()

    with requests.Session() as session:
        session.headers.update(HEADERS)
        
        while len(visited) < max_pages:
            # Pop from high priority queue first, then regular
            if high_priority_queue:
                url = high_priority_queue.popleft()
            elif regular_queue:
                url = regular_queue.popleft()
            else:
                break  # Both queues empty
                
            if url in visited:
                continue

            visited.add(url)
            try:
                response = session.get(url, timeout=20)
                response.raise_for_status()
            except requests.RequestException:
                continue

            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type:
                continue

            page, links = parse_page(response.text, url)
            
            # Skip pages with very little content (tag pages, empty pages)
            if len(page["content"]) < min_content_length:
                continue
            
            # Skip duplicate pages (same title & similar content)
            if is_duplicate_page(page, pages):
                continue

            pages.append(page)
            urls_seen.add(url)
            
            # Categorize links
            for link in links:
                if link not in visited and link not in urls_seen:
                    urls_seen.add(link)
                    if is_high_priority(link):
                        high_priority_queue.append(link)
                    elif not should_skip_url(link):
                        regular_queue.append(link)

            sleep(delay_seconds)

    return pages


def save_pages(pages: list[dict], path: Path = DATA_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(pages, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_pages(path: Path = DATA_PATH) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8-sig"))


if __name__ == "__main__":
    scraped_pages = scrape_site()
    save_pages(scraped_pages)
    print(f"Saved {len(scraped_pages)} unique pages to {DATA_PATH}")
    
    # Show summary of what was scraped
    print("\n=== Scraped Pages Summary ===")
    for i, page in enumerate(scraped_pages, 1):
        content_len = len(page.get("content", ""))
        print(f"{i:2d}. [{content_len:5d} chars] {page.get('title', 'N/A')[:60]}")
        print(f"     URL: {page.get('url', '')}")