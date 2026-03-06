from typing import Dict, Any
from agents.state import AgenticRAGState

# Import your existing hybrid retrieval logic
from src.hybrid_retrieval import retrieve_hybrid

class RegionalPolicyAgent:
    """
    A regional specialist node that retrieves country-specific financial policies.
    """
    def __init__(self, country: str, faiss_index, bm25_index, documents, embed_model):
        self.country = country
        self.faiss_index = faiss_index
        self.bm25_index = bm25_index
        self.documents = documents
        self.embed_model = embed_model # Needed for Stage 1 candidate generation

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the two-stage retrieval for the specific region.
        """
        query = state.get("query", "")
        print(f"[{self.country} Agent] Searching policy documents for: {query[:50]}...")

        # CRITICAL: We MUST use keyword arguments here to match your retriever's '*' signature.
        retrieved_chunks = retrieve_hybrid(
            query=query,
            documents=self.documents,
            bm25=self.bm25_index,
            embed_model=self.embed_model,
            faiss_index=self.faiss_index,
            top_k=8,
            faiss_k=40,
            bm25_k=40
        )

        # Format the context for the synthesizer
        context_str = "\n\n".join([
            f"Source: {c['metadata']['source_file']}\nContent: {c['text']}" 
            for c in retrieved_chunks
        ])

        # Update the state based on which country this agent is responsible for
        update = {}
        if self.country == "India":
            update["india_context"] = context_str
        elif self.country == "Singapore":
            update["sg_context"] = context_str
            
        return update