from typing import Dict, Any
from langchain_core.messages import AIMessage
# Assuming AgentState is the updated TypedDict in agents.state
from agents.state import AgentState
from src.hybrid_retrieval import retrieve_hybrid

class RegionalPolicyAgent:
    """
    A regional specialist node that retrieves country-specific financial policies
    and shares them with the team via the message state.
    """
    def __init__(self, country: str, faiss_index, bm25_index, documents, embed_model):
        self.country = country
        self.faiss_index = faiss_index
        self.bm25_index = bm25_index
        self.documents = documents
        self.embed_model = embed_model

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        # 1. Get query from the first HumanMessage in the conversation
        query = state["messages"][0].content
        
        print(f"[{self.country} Agent] Fetching {self.country} policy data for the team...")

        # Execute your existing high-performance hybrid retrieval
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

        # 2. Format the context into a clear, cited message for the team
        context_body = "\n\n".join([
            f"[Source: {c['metadata']['source_file']}]\n{c['text']}" 
            for c in retrieved_chunks
        ])
        
        reply_content = (
            f"I have retrieved the relevant {self.country} policy details:\n\n"
            f"{context_body}\n\n"
            f"@Quantitative_Engine, please use these rules for any necessary calculations."
        )

        # 3. Return the message and pass the baton
        return {
            "messages": [AIMessage(content=reply_content, name=f"{self.country}_Retriever")],
            "next_speaker": "Quantitative_Engine" 
        }