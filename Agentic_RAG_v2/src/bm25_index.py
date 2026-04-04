import re
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

def normalize_text(t: str) -> str:
    """Standardizes spacing and casing for consistent lexical matching."""
    t = t.lower()
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def tokenize(text: str) -> List[str]:
    """
    Custom Finance + Table-friendly tokenizer.
    Preserves percentages, decimals, and hyphenated financial terms.
    """
    text = normalize_text(text)
    # Regex captures: words, words with apostrophes, numbers with decimals/percentages, and hyphenated terms
    return re.findall(r"[a-zA-Z]+(?:'[a-z]+)?|\d+(?:\.\d+)?%?|\w+(?:-\w+)+", text)

def build_bm25(documents: List[Dict[str, Any]]) -> BM25Okapi:
    """Builds the Okapi BM25 index from the chunked documents."""
    tokenized = [tokenize(d["text"]) for d in documents]
    return BM25Okapi(tokenized)