import time
import pandas as pd
import json
import os
import re
import sys
from typing import List, Dict, Any

# 1. DYNAMIC PATH FIX (Ensures 'src' and 'agents' are findable)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.rag import init_llm
from src.embeddings import load_embedding_model
from src.country_indexes import load_country_indexes
from agents.graph import build_agentic_workflow
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

class RAGEvaluator:
    """
    Evaluates the Multi-Agent Financial System and records the full conversation track.
    """
    def __init__(self, api_key: str, india_indexes: Dict, sg_indexes: Dict):
        self.llm_fleet = {
            "fast_llm": init_llm(api_key=api_key, model="meta/llama3-8b-instruct"),
            "smart_llm": init_llm(api_key=api_key, model="meta/llama3-70b-instruct"),
            "writer_llm": init_llm(api_key=api_key, model="mistralai/mixtral-8x22b-instruct-v0.1")
        }
        self.app = build_agentic_workflow(self.llm_fleet, india_indexes, sg_indexes)
        self.judge_llm = self.llm_fleet["smart_llm"]

    def evaluate_response(self, query: str, expected_focus: str, actual_response: str) -> Dict[str, Any]:
        """LLM-as-a-Judge to score the final output on a 0-100 scale."""
        eval_prompt = f"""
        You are a Senior Financial Auditor. Grade this AI response on a strict scale of 0 to 100.
        Query: {query}
        Expected Focus: {expected_focus}
        AI Response: {actual_response}
        
        Rate the response out of 100 for:
        1. "accuracy": 100 if math/facts are exact, 0 if hallucinated.
        2. "completeness": 100 if it addressed all parts of the query.
        3. "citation": 100 if it cited specific sources properly.
        
        Return ONLY a JSON object: {{"accuracy": 85, "completeness": 90, "citation": 75, "reasoning": "string"}}
        """
        raw_eval = self.judge_llm(eval_prompt)
        try:
            clean_json = re.sub(r"```json|```", "", raw_eval).strip()
            return json.loads(clean_json)
        except:
            return {"accuracy": 0, "completeness": 0, "citation": 0, "reasoning": "Evaluation failed to parse."}

    def run_benchmark(self, test_queries: List[Dict[str, Any]]):
        full_results = []
        
        for i, test in enumerate(test_queries, 1):
            query = test["query"]
            focus = test["focus"]
            test_id = test.get("id", f"unknown_test_{i}")
            print(f"📊 Running Test {i}/{len(test_queries)}: {test_id}")

            start_time = time.time()
            inputs = {
                "messages": [HumanMessage(content=query, name="User")],
                "next_speaker": "Orchestrator"
            }
            
            # TRACKING THE CONVERSATION
            conversation_track = []
            final_answer = "TIMEOUT / NO ANSWER"
            
            try:
                # Stream the meeting
                for output in self.app.stream(inputs):
                    node_name = list(output.keys())[0]
                    node_state = output[node_name]
                    
                    if "messages" in node_state:
                        last_msg = node_state["messages"][-1]
                        # Log the turn (Agent-to-Agent conversation)
                        conversation_track.append({
                            "agent": last_msg.name or node_name,
                            "content": last_msg.content
                        })
                        
                        if node_name == "Synthesizer":
                            final_answer = last_msg.content
            except Exception as e:
                # If a graph routing error or hard crash occurs, log it and keep going!
                conversation_track.append({
                    "agent": "SYSTEM_ERROR",
                    "content": str(e)
                })
                final_answer = f"System crashed during execution: {str(e)}"

            exec_time = time.time() - start_time
            scores = self.evaluate_response(query, focus, final_answer)

            # Store full data object
            full_results.append({
                "id": test_id,
                "category": test.get("category", "General"),
                "metrics": {
                    "time_sec": round(exec_time, 2),
                    "accuracy": scores.get("accuracy", 0),
                    "completeness": scores.get("completeness", 0),
                    "citation": scores.get("citation", 0),
                    "turns": len(conversation_track)
                },
                "final_answer": final_answer,
                "conversation_history": conversation_track, # Full agent meeting transcript
                "judge_reasoning": scores.get("reasoning", "")
            })

        return full_results

if __name__ == "__main__":
    load_dotenv()
    key = os.getenv("NVIDIA_API_KEY")
    
    # Load data dependencies
    print("[System] Loading dependencies...")
    embed = load_embedding_model()
    idx = load_country_indexes(out_dir="data")
    i_idx = idx.get("India", {}); s_idx = idx.get("Singapore", {})
    i_idx["embed_model"] = embed; s_idx["embed_model"] = embed

    # Load golden dataset
    with open("evaluation/golden_dataset.json", "r") as f:
        test_suite = json.load(f)

    # Execute
    evaluator = RAGEvaluator(key, i_idx, s_idx)
    results = evaluator.run_benchmark(test_suite)

    # Save as JSON for full track history
    with open("evaluation/full_benchmark_track.json", "w") as f:
        json.dump(results, f, indent=4)
        
    # Print a quick summary table
    summary_df = pd.DataFrame([
        {
            "ID": r["id"], 
            "Acc(100)": r["metrics"]["accuracy"], 
            "Comp(100)": r["metrics"]["completeness"],
            "Time(s)": r["metrics"]["time_sec"], 
            "Turns": r["metrics"]["turns"]
        } for r in results
    ])
    print("\n--- BENCHMARK SUMMARY (0-100 SCALE) ---")
    print(summary_df.to_string(index=False))