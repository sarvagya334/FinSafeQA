import os
import json
import pickle
from typing import Dict, List, Any
import faiss

from src.embeddings import build_faiss_index
from src.bm25_index import build_bm25
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Prevent C++ thread collisions (resolves fork vs pthread deadlock risks)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _save_jsonl(path: str, items: List[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def build_country_indexes(documents: List[Dict[str, Any]], embed_model, out_dir: str, country_name: str) -> str:
    """
    Builds isolated FAISS and BM25 indexes for a specific country.
    """
    faiss_dir = os.path.join(out_dir, "faiss")
    bm25_dir = os.path.join(out_dir, "bm25")
    doc_dir = os.path.join(out_dir, "docmap")
    
    for d in [faiss_dir, bm25_dir, doc_dir]:
        _ensure_dir(d)

    if not documents:
        print(f"⚠️ No documents provided for {country_name}. Skipping.")
        return ""

    print(f"[Indexing] Building indices for {country_name} ({len(documents)} chunks)...")

    # 1. FAISS (Vector Store)
    texts = [d["text"] for d in documents]
    index, _ = build_faiss_index(texts, embed_model)
    faiss.write_index(index, os.path.join(faiss_dir, f"{country_name}.index"))

    # 2. BM25 (Keyword Search)
    bm25 = build_bm25(documents)
    with open(os.path.join(bm25_dir, f"{country_name}.pkl"), "wb") as f:
        pickle.dump(bm25, f)

    # 3. Document Map (Metadata)
    _save_jsonl(os.path.join(doc_dir, f"{country_name}.jsonl"), documents)
    
    return country_name

def load_country_indexes(out_dir: str) -> Dict[str, Dict[str, Any]]:
    """Loads all per-country indexes into memory for the LangGraph agents."""
    faiss_dir = os.path.join(out_dir, "faiss")
    bm25_dir = os.path.join(out_dir, "bm25")
    doc_dir = os.path.join(out_dir, "docmap")

    out = {}
    if not os.path.isdir(faiss_dir):
        return out

    for file in os.listdir(faiss_dir):
        if not file.endswith(".index"):
            continue
            
        country_key = file.replace(".index", "")
        
        try:
            index = faiss.read_index(os.path.join(faiss_dir, file))
            
            with open(os.path.join(bm25_dir, f"{country_key}.pkl"), "rb") as f:
                bm25 = pickle.load(f)
                
            docs = []
            with open(os.path.join(doc_dir, f"{country_key}.jsonl"), "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        docs.append(json.loads(line))
            
            out[country_key] = {
                "faiss_index": index,
                "bm25_index": bm25,
                "documents": docs
            }
        except Exception as e:
            print(f"Error loading indexes for {country_key}: {e}")

    return out