import os
import json
import hashlib
from typing import List, Dict, Any

def file_sha256(path: str) -> str:
    """Generates a SHA-256 hash to detect if a file has been modified."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def cache_path_for(md_path: str, cache_dir: str) -> str:
    """Determines the output path for the JSONL cache file."""
    base = os.path.basename(md_path)
    name = os.path.splitext(base)[0]
    return os.path.join(cache_dir, f"{name}.chunks.jsonl")

def write_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    """Writes dictionaries to disk as JSON Lines."""
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Loads JSON Lines from disk back into memory."""
    out = []
    if not os.path.exists(path):
        return out
        
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out