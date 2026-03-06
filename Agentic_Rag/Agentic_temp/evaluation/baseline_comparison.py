import time
import pandas as pd
import json
import os
import re
from typing import List, Dict, Any

# Import the existing LLM initializer
from src.rag import init_llm
from src.config import NVIDIA_BASE_URL
from dotenv import load_dotenv

# Import the New Agentic Architecture
from agents.graph import build_agentic_workflow
from langchain_core.messages import HumanMessage

# Import the Old Baseline
try:
    from src.multi_prompt_rag import multi_prompt_rag
except ImportError:
    print("Warning: Ensure multi_prompt_rag.py is accessible to run the baseline.")

class RAGEvaluator:
    """
    Empirically compares the Baseline Multi-Prompt RAG against the 
    LangGraph Multi-Agent RAG using an LLM-as-a-Judge approach.
    """
    def __init__(self, api_key: str, india_indexes: Dict, sg_indexes: Dict):
        self.llm = init_llm(api_key=api_key)
        self.app = build_agentic_workflow(self.llm, india_indexes, sg_indexes)
        
        # We need a unified retrieval function for the old baseline to use
        self.baseline_retriever = self._mock_baseline_retriever(india_indexes, sg_indexes)

    def _mock_baseline_retriever(self, india_idx: Dict, sg_idx: Dict):
        """Wraps the regional indexes so the old static RAG can query them blindly."""
        from src.hybrid_retrieval import retrieve_hybrid
        def retrieve(query: str, top_k: int = 5):
            # The old system didn't isolate domains, so it searches both blindly
            i_chunks = retrieve_hybrid(query, **india_idx, top_k=top_k)
            s_chunks = retrieve_hybrid(query, **sg_idx, top_k=top_k)
            return i_chunks + s_chunks
        return retrieve

    def evaluate_response(self, query: str, expected_focus: str, actual_response: str) -> Dict[str, Any]:
        """Uses the LLM to judge the quality, accuracy, and hallucination rate of the answer."""
        eval_prompt = f"""
        You are an academic evaluator grading a financial AI system.
        
        Query: {query}
        Expected Focus Areas: {expected_focus}
        
        AI Response:
        {actual_response}
        
        Rate the response strictly out of 10 on the following metrics:
        1. "accuracy": Did it calculate the numbers correctly without hallucinating?
        2. "completeness": Did it address both regions if comparative?
        3. "formatting": Did it cite sources properly?
        
        Output ONLY a JSON object: {{"accuracy": 8, "completeness": 9, "formatting": 7, "reasoning": "brief explanation"}}
        """
        raw_eval = self.llm(eval_prompt)
        try:
            clean_json = re.sub(r"```json|```", "", raw_eval).strip()
            return json.loads(clean_json)
        except json.JSONDecodeError:
            return {"accuracy": 0, "completeness": 0, "formatting": 0, "reasoning": "Evaluation Parsing Failed"}

    def run_benchmark(self, test_queries: List[Dict[str, str]]) -> pd.DataFrame:
        """Executes the tests and returns a pandas DataFrame for statistical analysis."""
        results = []
        total_tests = len(test_queries)

        for i, test in enumerate(test_queries, 1):
            print(f"\n--- Running Test {i}/{total_tests} ---")
            query = test.get("query", "")
            focus = test.get("focus", "")
            category = test.get("category", "General")
            
            # --- 1. Run Baseline ---
            print("Executing Baseline Multi-Prompt RAG...")
            start_time = time.time()
            try:
                base_ans = multi_prompt_rag(query, self.baseline_retriever, self.llm)
            except Exception as e:
                base_ans = f"Error: {str(e)}"
            base_time = time.time() - start_time
            base_scores = self.evaluate_response(query, focus, base_ans)

            # --- 2. Run Agentic LangGraph ---
            print("Executing Agentic LangGraph RAG...")
            start_time = time.time()
            try:
                inputs = {"query": query, "messages": [HumanMessage(content=query)]}
                final_state = self.app.invoke(inputs)
                agent_ans = final_state.get("final_answer", "Error: No answer")
                
                # Track tool usage
                tools_used = len(final_state.get("math_results", []))
            except Exception as e:
                agent_ans = f"Error: {str(e)}"
                tools_used = 0
            agent_time = time.time() - start_time
            agent_scores = self.evaluate_response(query, focus, agent_ans)

            # --- 3. Store Results ---
            results.append({
                "Category": category,
                "Query": query,
                "Baseline_Time(s)": round(base_time, 2),
                "Agentic_Time(s)": round(agent_time, 2),
                "Baseline_Accuracy": base_scores.get("accuracy", 0),
                "Agentic_Accuracy": agent_scores.get("accuracy", 0),
                "Agentic_Tools_Used": tools_used,
                "Agentic_Win": agent_scores.get("accuracy", 0) > base_scores.get("accuracy", 0),
                "Agentic_Reasoning": agent_scores.get("reasoning", "")
            })

        return pd.DataFrame(results)

if __name__ == "__main__":
    load_dotenv()
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("NVIDIA_API_KEY missing from environment variables.")
    
    # Load your indexes
    from src.country_indexes import load_country_indexes
    from src.embeddings import load_embedding_model
    
    embed_model = load_embedding_model()
    all_indexes = load_country_indexes(out_dir="data")
    
    india_idx = all_indexes.get("India", {})
    sg_idx = all_indexes.get("Singapore", {})
    india_idx["embed_model"] = embed_model
    sg_idx["embed_model"] = embed_model
    
    evaluator = RAGEvaluator(api_key, india_idx, sg_idx)
    
    # Load the Golden Dataset 
    dataset_path = "evaluation/golden_dataset.json"
    if os.path.exists(dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            test_suite = json.load(f)
        print(f"Loaded {len(test_suite)} golden queries for benchmarking.")
    else:
        print("Golden dataset not found. Falling back to default test suite.")
        test_suite = [
            {
                "category": "1_Direct_Policy",
                "query": "What is the penalty for withdrawing from an Indian 5-Year TD after 3 years?",
                "focus": "Exact penalty rules from Indian Post Office scheme PDFs."
            },
            {
                "category": "5_Real_Yield",
                "query": "Compare the real yield of an Indian 364-day T-Bill vs a 12-month Singapore T-Bill issued in Jan 2023.",
                "focus": "Must retrieve nominal rates and strictly apply mathematical inflation adjustments."
            },
            {
                "category": "3_Cross_Border_FX",
                "query": "If I convert S$20,000 to INR today, how much is it, and what is the maximum I can put into an SCSS account?",
                "focus": "Must use the FX Tool for currency conversion and verify the SCSS max limit."
            }
        ]
    
    # Run the benchmark and generate the report
    df_results = evaluator.run_benchmark(test_suite)
    
    print("\n=== EVALUATION REPORT ===")
    print(df_results.to_string())
    
    # Save the report
    report_path = "evaluation/benchmark_results.csv"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    df_results.to_csv(report_path, index=False)
    
    print(f"\nReport successfully saved to {report_path}. You can load this into scikit-learn or pandas for further variance analysis.")