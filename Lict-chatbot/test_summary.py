#!/usr/bin/env python
"""Test the summarization feature."""

import json
from pathlib import Path

# Test the chatbot functions
from chatbot import (
    is_summary_request,
    build_summary_answer,
    find_matching_pages,
)

# Load sample pages (from scraper module)
from scraper import load_pages

# Load sample pages
pages = load_pages(Path("data/lict_pages.json"))
print(f"Loaded {len(pages)} pages from data/lict_pages.json\n")

# Test summary detection
test_queries = [
    "summarize CSIT course",
    "what is BCA?",
    "give me an overview of BHM",
    "summarise the BIM program",
    "key points of BSc CSIT",
    "what are the courses offered at LICT?",
    "tell me about admission requirements",
]

print("=== Testing Summary Detection ===")
for query in test_queries:
    is_summary = is_summary_request(query)
    print(f"Query: '{query}'")
    print(f"  -> Summary request: {is_summary}\n")

# Test summarization with a query
print("\n=== Testing Summarization ===")
query = "summarize the CSIT course"
matches = find_matching_pages(query, pages, limit=3)
print(f"Found {len(matches)} matching pages for: '{query}'\n")

if matches:
    summary = build_summary_answer(matches)
    print("Generated Summary:")
    print("-" * 50)
    print(summary[:500] + "..." if len(summary) > 500 else summary)
