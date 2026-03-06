import os
import json
from typing import List, Dict, Any
from tqdm import tqdm

from src.config import (
    DATA_INDIA_RAW, 
    DATA_SG_RAW, 
    DATA_MACRO_RAW, 
    DATA_PROCESSED, 
    BASE_DIR
)
from pipelines.convert_docling import convert_all_raw_to_markdown
from src.chunk_cache import file_sha256
from src.chunking import chunk_markdown_page_level
from src.embeddings import load_embedding_model
from src.country_indexes import build_country_indexes
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"

def infer_metadata_from_path(file_path: str) -> Dict[str, Any]:
    """
    Infers the region and asset class strictly based on the directory structure.
    This prevents the LLM from getting confused about where data originated.
    """
    meta = {
        "source_file": os.path.basename(file_path),
        "source_type": "markdown",
        "country": "Unknown",
        "asset_class": "Unknown"
    }
    
    path_lower = file_path.lower()
    
    # 1. Determine Country
    if "india_stable_assets" in path_lower:
        meta["country"] = "India"
    elif "singapore_stable_assets" in path_lower:
        meta["country"] = "Singapore"
    elif "macro_and_fx" in path_lower:
        # Macro data often applies to both or compares them
        meta["country"] = "Macro_Global" 
        meta["countries"] = ["India", "Singapore"]

    # 2. Determine Asset Class
    if "macro_and_fx" in path_lower:
        meta["asset_class"] = "Macro"
    elif "stable_assets" in path_lower:
        meta["asset_class"] = "Stable"
        
    return meta

def process_markdown_file(md_path: str) -> List[Dict[str, Any]]:
    """Reads a converted markdown file, chunks it, and applies metadata."""
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    sha = file_sha256(md_path)
    base_meta = infer_metadata_from_path(md_path)
    base_meta["file_sha256"] = sha

    # Use your existing excellent page-level chunker
    raw_chunks = chunk_markdown_page_level(md_text)

    docs = []
    for i, text in enumerate(raw_chunks, 1):
        # We append the metadata directly into the chunk dictionary 
        # so FAISS/BM25 can filter it later.
        docs.append({
            "chunk_id": f"{base_meta['source_file']}_chunk_{i}",
            "text": text,
            "metadata": base_meta.copy()
        })
        
    return docs

def run_ingestion_pipeline():
    print("[Ingestion] Starting pipeline...")

    # Initialize separate buckets for each regional index
    india_docs = []
    sg_docs = []
    macro_docs = []
    
    # 1. Process India Stable Assets
    print("\n[Ingestion] Processing India Stable Assets...")
    india_mds = convert_all_raw_to_markdown(DATA_INDIA_RAW, DATA_PROCESSED)
    for md in tqdm(india_mds, desc="Chunking India Files", unit="file"):
        india_docs.extend(process_markdown_file(md))

    # 2. Process Singapore Stable Assets
    print("\n[Ingestion] Processing Singapore Stable Assets...")
    sg_mds = convert_all_raw_to_markdown(DATA_SG_RAW, DATA_PROCESSED)
    for md in tqdm(sg_mds, desc="Chunking SG Files", unit="file"):
        sg_docs.extend(process_markdown_file(md))

    # 3. Process Macro & FX 
    print("\n[Ingestion] Processing Macro & FX Data...")
    macro_mds = convert_all_raw_to_markdown(DATA_MACRO_RAW, DATA_PROCESSED)
    for md in tqdm(macro_mds, desc="Chunking Macro Files", unit="file"):
        macro_docs.extend(process_markdown_file(md))

    # 4. Load Model once to save memory
    print("\n[Ingestion] Loading embedding model...")
    embed_model = load_embedding_model()
    out_dir = os.path.join(BASE_DIR, "data")

    # 5. Build Indexes SEPARATELY
    # This ensures main.py can find 'India.index' and 'Singapore.index'
    print("[Ingestion] Building isolated country vector spaces...")
    
    # Process India
    if india_docs:
        build_country_indexes(india_docs, embed_model, out_dir, country_name="India")
    
    # Process Singapore
    if sg_docs:
        build_country_indexes(sg_docs, embed_model, out_dir, country_name="Singapore")

    # Process Macro (Optional: You can merge macro into both if needed)
    if macro_docs:
        build_country_indexes(macro_docs, embed_model, out_dir, country_name="Macro")

    print(f"\n✅ [Ingestion] Success! Individual indices created in {out_dir}")

if __name__ == "__main__":
    run_ingestion_pipeline()