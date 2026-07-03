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
        "Mozilla/5.0 (compatible; LICTChatbotScraper/1.0; "
        "+https://lict.edu.np/)"
    )
}


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


def parse_page(html: str, url: str) -> tuple[dict, list[str]]:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "iframe", "form"]):
        tag.decompose()

    title_tag = soup.find(["h1", "h2"]) or soup.find("title")
    title = clean_text(title_tag.get_text(" ")) if title_tag else url

    content_root = soup.find("main") or soup.find("article") or soup.body or soup
    text_parts = []
    for tag in content_root.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "th"]):
        value = clean_text(tag.get_text(" "))
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
        if is_same_site(normalized) and not re.search(r"\.(jpg|jpeg|png|gif|pdf|zip|rar)$", normalized, re.I):
            links.append(normalized)

    return page, links


def scrape_site(
    start_url: str = BASE_URL,
    max_pages: int = 40,
    delay_seconds: float = 0.4,
) -> list[dict]:
    start_url = normalize_url(start_url)
    queue = deque([start_url])
    visited = set()
    pages = []

    with requests.Session() as session:
        session.headers.update(HEADERS)
        while queue and len(visited) < max_pages:
            url = queue.popleft()
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
            if len(page["content"]) > 80:
                pages.append(page)

            for link in links:
                if link not in visited and link not in queue:
                    queue.append(link)

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
    print(f"Saved {len(scraped_pages)} pages to {DATA_PATH}")
