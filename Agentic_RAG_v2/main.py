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
    # 1. Load Environment Variables
    load_dotenv()
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("Please set NVIDIA_API_KEY in your .env file.")

    # 2. Initialize the LLM Fleet
    print("[System] Initializing the LLM Fleet...")
    llm_fleet = {
        # The Orchestrator is just routing traffic; it needs speed, not deep math.
        "fast_llm": init_llm(api_key=api_key, model="meta/llama3-8b-instruct"),
        
        # The Quant and Critic need massive reasoning capabilities to avoid hallucinating.
        "smart_llm": init_llm(api_key=api_key, model="meta/llama3-70b-instruct"),
        
        # The Synthesizer just needs to write good prose and format markdown well.
        "writer_llm": init_llm(api_key=api_key, model="mistralai/mixtral-8x22b-instruct-v0.1")
    }

    print("[System] Loading Embedding Models...")
    embed_model = load_embedding_model()

    # 3. Load the Pre-built FAISS & BM25 Indexes
    print("[System] Loading Regional Vector Indexes...")
    all_indexes = load_country_indexes(out_dir="data") 
    
    india_indexes = all_indexes.get("India")
    sg_indexes = all_indexes.get("Singapore")
    
    if not india_indexes or not sg_indexes:
        print("[Warning] Indexes not found. Please run pipelines/ingestion.py first.")
        return

    # Pass the embedding model into the index dictionaries
    india_indexes["embed_model"] = embed_model
    sg_indexes["embed_model"] = embed_model

    # 4. Compile the Multi-Agent LangGraph
    print("[System] Compiling Multi-Agent State Machine...")
    # Pass the entire fleet instead of a single LLM
    app = build_agentic_workflow(llm_fleet, india_indexes, sg_indexes)

    # 5. Execute a Test Query
    test_query = (
        "give a list of all historical interesst rate of T-bills in india"
    )
    
    print(f"\n[User Query] {test_query}\n")
    print("--- Execution Trace ---")
    
    # Initialize the AgentState with standard conversational messages
    inputs = {
        "messages": [HumanMessage(content=test_query, name="User")],
        "next_speaker": "Orchestrator" # Jump-start the conversation
    }
    
    # Stream the graph execution node by node
    for output in app.stream(inputs):
        node_name = list(output.keys())[0]
        node_state = output[node_name]
        
        print(f"\n✅ Finished Node: {node_name}")
        
        # Print the actual conversation messages logged by the agents
        if "messages" in node_state:
            # Grab the last message added by the agent
            latest_msg = node_state["messages"][-1]
            print(f"   💬 [{latest_msg.name}]: {latest_msg.content[:150]}...") 

if __name__ == "__main__":
    main()