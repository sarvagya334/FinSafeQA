import os

# ---- Core Paths ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_INDIA_RAW = os.path.join(DATA_RAW, "india_stable_assets")
DATA_SG_RAW = os.path.join(DATA_RAW, "singapore_stable_assets")
DATA_MACRO_RAW = os.path.join(DATA_RAW, "macro_and_fx")

DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
DATA_FAISS = os.path.join(BASE_DIR, "data", "faiss")
DATA_BM25 = os.path.join(BASE_DIR, "data", "bm25")
DATA_DOCMAP = os.path.join(BASE_DIR, "data", "docmap")

# Create folders if missing
for p in [DATA_INDIA_RAW, DATA_SG_RAW, DATA_MACRO_RAW, DATA_PROCESSED, DATA_FAISS, DATA_BM25, DATA_DOCMAP]:
    os.makedirs(p, exist_ok=True)

# ---- Metadata Rules ----
VALID_COUNTRIES = {"India", "Singapore"}
VALID_ASSET_CLASSES = {"Stable", "Growth", "Macro"}
AMBIGUOUS_TERMS = ["sgs", "td", "bond"]

# ---- Models ----
EMBEDDING_MODEL_NAME = "mukaj/fin-mpnet-base"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2" # NEW: For re-ranking

# ---- LLM Config ----
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "meta/llama3-70b-instruct" # Upgraded for agentic reasoning
DEFAULT_TEMPERATURE = 0.1