import json
import re
from typing import Dict, Any, List
from agents.state import AgenticRAGState

class OrchestratorAgent:
    """
    The Supervisor node that analyzes the user's query and determines the 
    execution graph path. Enforces strict JSON output for LangGraph conditional edges.
    """
    def __init__(self, llm_callable):
        # llm_callable is your existing init_llm() from rag.py
        self.llm = llm_callable

    def __call__(self, state: AgenticRAGState) -> Dict[str, Any]:
        query = state["query"]

        prompt = f"""
        You are the Supervisor Agent for a Cross-Border Financial RAG System.
        Analyze the following user query and determine which specialized agents must be triggered.
        
        Query: '{query}'
        
        Available Agents:
        - "India_Policy_Agent": Trigger if the query mentions Indian assets (T-Bills, PPF, SCSS, etc.), INR, or Indian taxation.
        - "Singapore_Policy_Agent": Trigger if the query mentions Singaporean assets (SSBs, CPF, SGS, etc.), SGD, or Singaporean taxation.
        - "Quantitative_Agent": Trigger if the query requires yield calculation, compound interest, inflation adjustment, or currency conversion.

        Respond ONLY with a strict JSON dictionary containing boolean values. Do not include markdown formatting or explanations.
        
        Example Output:
        {{
            "India_Policy_Agent": true,
            "Singapore_Policy_Agent": true,
            "Quantitative_Agent": true
        }}
        """
        
        raw_response = self.llm(prompt)
        
        try:
            # Clean potential markdown backticks from LLM output
            clean_json = re.sub(r"```json|```", "", raw_response).strip()
            routing_decisions = json.loads(clean_json)
        except json.JSONDecodeError:
            # Deterministic fallback if the LLM fails to output valid JSON
            routing_decisions = {
                "India_Policy_Agent": "india" in query.lower() or "inr" in query.lower(),
                "Singapore_Policy_Agent": "singapore" in query.lower() or "sgd" in query.lower() or "cpf" in query.lower(),
                "Quantitative_Agent": any(word in query.lower() for word in ["calculate", "compare", "yield", "return", "tax"])
            }
            
            # If everything is false in fallback, default to triggering both regional agents
            if not any(routing_decisions.values()):
                routing_decisions["India_Policy_Agent"] = True
                routing_decisions["Singapore_Policy_Agent"] = True

        # We append a system message to the state log to track what the Orchestrator decided
        decision_log = f"Orchestrator routed to: {', '.join([k for k, v in routing_decisions.items() if v])}"
        
        return {
            # Note: LangGraph conditional edges will read this hidden state to branch the execution
            "routing_decisions": routing_decisions,
            "messages": [("system", decision_log)]
        }