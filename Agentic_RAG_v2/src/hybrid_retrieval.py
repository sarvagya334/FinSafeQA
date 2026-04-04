from typing import List, Dict, Any
import numpy as np
from sentence_transformers import CrossEncoder
from src.bm25_index import tokenize
from src.config import CROSS_ENCODER_MODEL_NAME

# Initialize the reranker globally to avoid reloading it on every query
try:
    reranker = CrossEncoder(CROSS_ENCODER_MODEL_NAME, max_length=512)
except Exception as e:
    print(f"Warning: Could not load CrossEncoder: {e}")
    reranker = None

def retrieve_hybrid(
    query: str,
    *,
    documents: List[Dict[str, Any]],
    bm25,
    embed_model,
    faiss_index,
    top_k: int = 8,
    faiss_k: int = 40,
    bm25_k: int = 40,
) -> List[Dict[str, Any]]:
    """
    Two-Stage Retrieval:
    1. Fast Candidate Generation (FAISS + BM25)
    2. Deep Cross-Encoder Re-ranking
    """
    if not documents:
        return []

    # --- Stage 1: Fast Candidate Generation ---
    # FAISS (Dense)
    qvec = embed_model.encode([query], normalize_embeddings=True).astype("float32")
    D, I = faiss_index.search(qvec, faiss_k)
    faiss_candidates = {int(idx) for idx in I[0] if idx >= 0 and int(idx) < len(documents)}

    # BM25 (Sparse/Lexical)
    qtok = tokenize(query)
    bm_scores_all = bm25.get_scores(qtok)
    bm_sorted = np.argsort(bm_scores_all)[::-1][:bm25_k]
    bm25_candidates = {int(i) for i in bm_sorted if 0 <= int(i) < len(documents) and bm_scores_all[int(i)] > 0}

    # Union of candidates (Broad Net)
    unique_candidates = list(faiss_candidates | bm25_candidates)
    if not unique_candidates:
        return []

    candidate_docs = [documents[i] for i in unique_candidates]

    # --- Stage 2: Cross-Encoder Re-ranking ---
    if reranker:
        # Create pairs of (Query, Document Text)
        pairs = [[query, doc["text"]] for doc in candidate_docs]
        
        # Predict semantic relevance scores
        cross_scores = reranker.predict(pairs)
        
        # Sort documents by the new Cross-Encoder score
        ranked_indices = np.argsort(cross_scores)[::-1]
        ranked_docs = [candidate_docs[i] for i in ranked_indices]
    else:
        # Fallback if reranker fails to load
        ranked_docs = candidate_docs

    # --- Diversity by File (Prevent one PDF from dominating) ---
    out = []
    file_count = {}
    per_file_limit = 3

    for doc in ranked_docs:
        src = doc.get("metadata", {}).get("source_file", "unknown")
        if file_count.get(src, 0) < per_file_limit:
            out.append(doc)
            file_count[src] = file_count.get(src, 0) + 1
            if len(out) >= top_k:
                break

    return out