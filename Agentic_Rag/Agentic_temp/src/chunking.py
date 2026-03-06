import re
from typing import List, Tuple

def approx_tokens(text: str) -> int:
    """Estimates token count, penalizing markdown tables due to pipe characters."""
    base = max(1, len(text) // 4)
    base += text.count("|") // 6
    return base

def split_by_pages(md: str) -> List[Tuple[int, str]]:
    """Splits the Docling markdown into distinct pages using form feeds or headers."""
    md = md.replace("\r\n", "\n")
    pages = []
    
    # Docling often uses \f (form feed) for page breaks
    parts = md.split("\f")
    for i, p in enumerate(parts, 1):
        if p.strip():
            pages.append((i, p.strip()))
            
    if not pages:
        pages = [(1, md)]
    return pages

def chunk_text(text: str, max_tokens: int = 400, overlap: int = 50) -> List[str]:
    """Basic sliding-window chunking for standard paragraphs."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        # Approximate 1 token = 0.75 words
        chunk_words = words[i:i + int(max_tokens * 0.75)]
        chunks.append(" ".join(chunk_words))
        i += int((max_tokens - overlap) * 0.75)
    return chunks

def chunk_markdown_page_level(md: str, max_tokens: int = 400) -> List[str]:
    """
    Primary chunking function. Splits by page, then breaks down into semantic chunks 
    while preserving page context in the prefix.
    """
    final_chunks = []
    pages = split_by_pages(md)

    for page_no, page_md in pages:
        page_prefix = f"[PAGE: {page_no}]\n"
        
        # Split by markdown headers (H1-H3)
        sections = re.split(r"(^#{1,3}\s+.*)", page_md, flags=re.MULTILINE)
        
        current_section = ""
        for part in sections:
            part = part.strip()
            if not part:
                continue
                
            if re.match(r"^#{1,3}\s+", part):
                # If we hit a new header, chunk the old section and start a new one
                if current_section:
                    for chunk in chunk_text(current_section, max_tokens):
                        final_chunks.append(page_prefix + chunk)
                current_section = part + "\n"
            else:
                current_section += part + "\n"
                
        # Catch the last section
        if current_section:
            for chunk in chunk_text(current_section, max_tokens):
                final_chunks.append(page_prefix + chunk)

    return [c.strip() for c in final_chunks if c.strip()]