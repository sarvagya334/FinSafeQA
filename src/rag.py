from typing import List, Dict, Any, Optional, Callable
import os
import requests

from src.hybrid_retrieval import retrieve_hybrid
from src.country_detect import detect_country_from_query


# -----------------------------
# LLM INIT
# -----------------------------

def init_llm_ollama(
    model: str = "llama3.2",
    base_url: str = "http://localhost:11434",
    timeout: int = 120,
) -> Callable[[str], str]:
    """
    Local LLM via Ollama — runs on Apple Silicon via Metal (MPS).
    Install:  brew install ollama && ollama pull llama3.2
    Start:    ollama serve   (or it auto-starts on macOS)
    """
    endpoint = f"{base_url.rstrip('/')}/api/chat"

    def llm(prompt: str) -> str:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a careful assistant that only answers from provided context."},
                {"role": "user",   "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 512},
        }
        r = requests.post(endpoint, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()["message"]["content"]

    return llm


def init_llm_nvidia(
    api_key: str,
    model: str = "meta/llama-3.1-8b-instruct",
    timeout: int = 180,   # 180s — large models (405B, 253B) can take 90-120s
) -> Callable[[str], str]:
    """NVIDIA NIM cloud endpoint."""
    base_url = os.environ.get("NVIDIA_BASE_URL", "").strip()
    if not base_url:
        base_url = "https://integrate.api.nvidia.com/v1/chat/completions"
    elif not base_url.endswith("/chat/completions"):
        base_url = base_url.rstrip("/") + "/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    def llm(prompt: str) -> str:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a careful assistant that only answers from provided context."},
                {"role": "user",   "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 512,
        }
        r = requests.post(base_url, headers=headers, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    return llm


def init_llm(
    api_key: str | None = None,
    model: str | None = None,
    backend: str | None = None,
) -> Callable[[str], str]:
    """
    Factory that selects the LLM backend.

    Priority:
      1. `backend` argument
      2. LLM_BACKEND env var  ("ollama" | "nvidia")
      3. Auto-detect: Ollama if reachable, else NVIDIA

    Models default:
      ollama  → llama3.2
      nvidia  → meta/llama-3.1-8b-instruct
    """
    backend = (backend or os.environ.get("LLM_BACKEND", "")).strip().lower()

    if not backend:
        # Auto-detect: try Ollama first (no API key needed, local, fast)
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=3)
            if r.status_code == 200:
                backend = "ollama"
        except Exception:
            pass
        if not backend:
            backend = "nvidia"

    if backend == "ollama":
        ollama_model = model or os.environ.get("OLLAMA_MODEL", "llama3.2")
        print(f"[LLM] Using Ollama backend → model: {ollama_model}")
        return init_llm_ollama(model=ollama_model)

    # nvidia fallback
    if not api_key:
        api_key = os.environ.get("NVIDIA_API_KEY", "")
    nvidia_model = model or os.environ.get("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct")
    print(f"[LLM] Using NVIDIA backend → model: {nvidia_model}")
    return init_llm_nvidia(api_key=api_key, model=nvidia_model)



# -----------------------------
# Helpers
# -----------------------------

def _format_context(chunks: List[Dict[str, Any]], max_chars: int = 14000) -> str:
    parts = []
    total = 0
    for i, ch in enumerate(chunks, start=1):
        meta = ch.get("metadata", {})
        src = meta.get("source_file", "unknown")
        cid = meta.get("chunk_id", "?")
        block = f"[{i}] source={src}, chunk={cid}\n{ch['text']}".strip()
        total += len(block)
        if total > max_chars:
            break
        parts.append(block)
    return "\n\n".join(parts)


def _group_by_source(chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for ch in chunks:
        src = ch.get("metadata", {}).get("source_file", "unknown")
        grouped.setdefault(src, []).append(ch)
    return grouped


# -----------------------------
# Main RAG Query (country-aware)
# -----------------------------

def retrieve_and_build_prompt(
    *,
    query: str,
    country_indexes: Dict[str, Dict[str, Any]],
    embed_model,
) -> Optional[str]:
    """
    Retrieval + prompt construction. Returns a context-based prompt 
    OR a general knowledge fallback prompt if no documents are found.
    """
    requested = detect_country_from_query(query)

    def retrieve_from_pack(pack, q: str, top_k: int = 18):
        return retrieve_hybrid(
            q,
            documents=pack["docs"],
            bm25=pack["bm25"],
            embed_model=embed_model,
            faiss_index=pack["faiss"],
            top_k=top_k,
            alpha=0.65,
            faiss_k=100,
            bm25_k=160,
            per_file_limit=4,
        )

    # 1) Attempt Retrieval
    chunks = []
    if requested and requested in country_indexes:
        chunks = retrieve_from_pack(country_indexes[requested], query, top_k=18)
    elif not requested:
        merged = []
        for _, pack in country_indexes.items():
            merged.extend(retrieve_from_pack(pack, query, top_k=8))
        merged.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)
        chunks = merged[:20]

    # 2) Handle "Summarize All" sentinel
    q_lower = query.lower()
    if any(x in q_lower for x in ["summarize all documents", "summarize everything"]):
        return "__SUMMARIZE_ALL__"

    # 3) Build Prompt: Context-Aware vs. Generative Fallback
    if chunks:
        ctx = _format_context(chunks)
        return (
            f"You are a strict RAG assistant. Answer ONLY using the context provided.\n"
            f"If the answer isn't present, say \"Not found in documents.\"\n\n"
            f"Question: {query}\n\n"
            f"Context:\n{ctx}\n\n"
            f"Answer in a structured format with key numbers cited."
        )
    else:
        # FALLBACK: Generative mode when no internal documents match
        return (
            f"You are a financial research assistant. No specific internal documents "
            f"were found matching this query.\n\n"
            f"Please answer the following question using your general expert knowledge "
            f"of global financial markets and regulations. Explicitly state that "
            f"this information is based on general knowledge and not internal records.\n\n"
            f"Question: {query}"
        )


def rag_query(
    *,
    query: str,
    country_indexes: Dict[str, Dict[str, Any]],
    embed_model,
    llm,
) -> str:
    """
    Updated RAG Query that falls back to LLM general knowledge if 
    retrieval returns no relevant evidence.
    """
    requested = detect_country_from_query(query)

    def retrieve_from_pack(pack, q: str, top_k: int = 18):
        return retrieve_hybrid(
            q,
            documents=pack["docs"],
            bm25=pack["bm25"],
            embed_model=embed_model,
            faiss_index=pack["faiss"],
            top_k=top_k,
            alpha=0.65,
            faiss_k=100,
            bm25_k=160,
            per_file_limit=4,
        )

    # 1) Handle specific country or merged search
    chunks = []
    if requested:
        if requested in country_indexes:
            chunks = retrieve_from_pack(country_indexes[requested], query, top_k=18)
    else:
        merged = []
        for _, pack in country_indexes.items():
            merged.extend(retrieve_from_pack(pack, query, top_k=8))
        merged.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)
        chunks = merged[:20]

    # 2) Check for "Summarize All" mode
    q_lower = query.lower()
    summarize_all = any(x in q_lower for x in ["summarize all", "summary of all"])

    if summarize_all and chunks:
        grouped = _group_by_source(chunks)
        per_doc_summaries = []
        for src, src_chunks in grouped.items():
            ctx = _format_context(src_chunks, max_chars=9000)
            prompt = f"Summarize document: {src}\nContext:\n{ctx}\n\nBullet points with key rates/limits."
            per_doc_summaries.append(f"### {src}\n{llm(prompt).strip()}")
        
        merge_prompt = f"Consolidate these summaries for query: {query}\n\n{chr(10).join(per_doc_summaries)}"
        return llm(merge_prompt).strip()

    # 3) Final Output Logic
    if chunks:
        ctx = _format_context(chunks)
        prompt = (
            f"You are a strict RAG assistant. Answer ONLY using the provided context.\n\n"
            f"Question: {query}\n\n"
            f"Context:\n{ctx}\n\n"
            f"Answer clearly and include numbers if present."
        )
    else:
        # GENERATIVE FALLBACK
        prompt = (
            f"You are a financial research expert. No internal documents were found "
            f"matching this query. Answer the question below using your general "
            f"knowledge. Start with: 'Note: No matching internal records found.'\n\n"
            f"Question: {query}"
        )
    
    return llm(prompt).strip()