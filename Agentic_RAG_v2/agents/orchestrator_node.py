import json
import re
from typing import Dict, Any
from agents.state import AgentState

# TO THIS:
from agents.state import AgentState
from langchain_core.messages import AIMessage
# Import the unified state
from agents.state import AgentState 

class OrchestratorAgent:
    """
    The Supervisor node. Analyzes the user's query and "tags" the 
    first agent to begin the financial analysis.
    """
    def __init__(self, llm):
        self.llm = llm

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        # 1. Get the original user query from the first message
        user_query = state["messages"][0].content

        prompt = f"""
        You are the Orchestrator for a Cross-Border Financial Analysis Team.
        Review the User's Request: '{user_query}'
        
        Determine the BEST agent to start this task.
        
        Available Agents:
        - "India_Policy": For Indian assets (PPF, SCSS, SGB), INR, or Indian tax rules.
        - "Singapore_Policy": For Singapore assets (CPF, SSB, SGS), SGD, or Singapore tax rules.
        - "Quantitative_Engine": For macro SQL queries (Repo rates, CPI, SORA) or immediate math/FX.
        - "Developer_Engine": For requests involving charts, trends, or plotting data.

        Respond ONLY in strict JSON format.
        
        Example Output:
        {{
            "analysis": "User wants to compare India and SG yields, need India policy first.",
            "next_speaker": "India_Policy"
        }}
        """
        
        raw_response = self.llm(prompt)
        
        try:
            clean_json = re.sub(r"```json|```", "", raw_response).strip()
            decision = json.loads(clean_json)
            next_agent = decision.get("next_speaker", "India_Policy") # Fallback
            analysis = decision.get("analysis", "Starting analysis.")
        except:
            # Deterministic Fallback if LLM fails
            next_agent = "India_Policy" if "india" in user_query.lower() else "Singapore_Policy"
            analysis = "Fallback routing engaged."

        # 2. Add a message to the chat so other agents know the plan
        orchestrator_msg = AIMessage(
            content=f"Task Analysis: {analysis}. @{next_agent}, please begin the data gathering.",
            name="Orchestrator"
        )
        
        return {
            "messages": [orchestrator_msg],
            "next_speaker": next_agent # This triggers the Universal Router in graph.py
        }