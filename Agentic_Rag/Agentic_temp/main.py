import os
import json
from dotenv import load_dotenv

# Import your existing configurations and index loaders
from src.country_indexes import load_country_indexes
from src.embeddings import load_embedding_model
from src.rag import init_llm

# Import the LangGraph compiler
from agents.graph import build_agentic_workflow
from langchain_core.messages import HumanMessage
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"
def main():
    # 1. Load Environment Variables (NVIDIA API Key)
    load_dotenv()
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("Please set NVIDIA_API_KEY in your .env file.")

    # 2. Initialize the Core Engine Components
    print("[System] Initializing LLM and Embedding Models...")
    llm = init_llm(api_key=api_key)
    embed_model = load_embedding_model()

    # 3. Load the Pre-built FAISS & BM25 Indexes
    print("[System] Loading Regional Vector Indexes...")
    all_indexes = load_country_indexes(out_dir="data") 
    
    india_indexes = all_indexes.get("India")
    sg_indexes = all_indexes.get("Singapore")
    
    if not india_indexes or not sg_indexes:
        print("[Warning] Indexes not found. Please run pipelines/ingestion.py first.")
        return

    # Pass the embedding model into the index dictionaries so agents can encode queries
    india_indexes["embed_model"] = embed_model
    sg_indexes["embed_model"] = embed_model

    # 4. Compile the Multi-Agent LangGraph
    print("[System] Compiling Multi-Agent State Machine...")
    app = build_agentic_workflow(llm, india_indexes, sg_indexes)

    # 5. Execute a Test Query
    test_query = (
        "give a list of all historical interesst rate of T-bills in india"
    )

    
    print(f"\n[User Query] {test_query}\n")
    print("--- Execution Trace ---")
    
    # Initialize the AgentState
    inputs = {"query": test_query, "messages": [HumanMessage(content=test_query)]}
    final_answer = "Error: No answer generated."
    
    # Stream the graph execution node by node
    for output in app.stream(inputs):
        # Identify which node just finished
        node_name = list(output.keys())[0]
        node_state = output[node_name]
        
        print(f"\n✅ Finished Node: {node_name}")
        
        # Print internal system messages logged by the agents
        if "messages" in node_state:
            latest_msg = node_state["messages"][-1]
            if isinstance(latest_msg, tuple) and latest_msg[0] == "system":
                print(f"   Log: {latest_msg[1]}")

        # Capture the final answer when the Synthesizer node completes
        if "final_answer" in node_state:
            final_answer = node_state["final_answer"]

    # 6. Print Final Answer
    print("\n" + "="*50)
    print("FINAL SYNTHESIZED RESPONSE")
    print("="*50)
    print(final_answer)

if __name__ == "__main__":
    main()