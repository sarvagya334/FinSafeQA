import os

# 1. Define the strict environment rules FIRST
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"

# 2. THEN import the heavy C++ backed ML libraries
import faiss
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL_NAME

def detect_device() -> str:
    """Safely detects hardware acceleration to prevent runtime crashes."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        # Force CPU on Mac to avoid MPS compatibility issues with certain FAISS operations
        return "cpu"     
    except ImportError:
        return "cpu"

def load_embedding_model() -> SentenceTransformer:
    """Loads the specialized financial embedding model."""
    device = detect_device()
    print(f"[Embeddings] Initializing {EMBEDDING_MODEL_NAME} on {device}")
    return SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)

def build_faiss_index(texts: list[str], model: SentenceTransformer):
    """
    Constructs a normalized Inner Product (Cosine Similarity) FAISS index.
    """
    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True
    ).astype("float32")

    # Normalize vectors for Cosine Similarity
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    
    # Inner Product on normalized L2 vectors equals Cosine Similarity
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index, embeddings