import json
import re
from typing import Dict, Any
from agents.state import AgenticRAGState
from mcp_tools.math_repl import finance_calculator
from mcp_tools.triangular_fx_tool import fx_tool

class QuantitativeAgent:
    """
    Extracts numerical parameters from the retrieved policy contexts and uses 
    MCP tools (Math REPL, FX Tool) to compute exact financial comparisons.
    """
    def __init__(self, llm_callable):
        self.llm = llm_callable

    def __call__(self, state: AgenticRAGState) -> Dict[str, Any]:
        query = state["query"]
        india_ctx = state.get("india_context", "")
        sg_ctx = state.get("sg_context", "")
        
        prompt = f"""
        You are the Quantitative Engine for a financial RAG system.
        Based on the user query and the retrieved contexts, generate a JSON object containing mathematical expressions or FX conversion requests.
        
        Query: {query}
        
        India Context:
        {india_ctx}
        
        Singapore Context:
        {sg_ctx}
        
        Rules for Output:
        1. Only use operators +, -, *, /, **, and numbers for "math_expression".
        2. If currency conversion is needed, fill out "fx_request" with "amount" and "target_date" (YYYY-MM-DD).
        3. Respond ONLY in valid JSON.
        
        Example:
        {{
            "math_expression": "100000 * (1 + 0.071)**3",
            "fx_request": {{"amount": 50000, "target_date": "2024-01-01"}}
        }}
        """
        
        raw_response = self.llm(prompt)
        math_results = []
        expressions_used = []
        
        try:
            clean_json = re.sub(r"```json|```", "", raw_response).strip()
            instructions = json.loads(clean_json)
            
            # Execute Math
            if "math_expression" in instructions and instructions["math_expression"]:
                expr = instructions["math_expression"]
                expressions_used.append(expr)
                res = finance_calculator.execute(expr)
                math_results.append(res)
                
            # Execute FX
            if "fx_request" in instructions and instructions["fx_request"]:
                fx = instructions["fx_request"]
                expressions_used.append(f"FX Convert: {fx['amount']} SGD on {fx['target_date']}")
                res = fx_tool.convert_sgd_to_inr(amount=fx['amount'], target_date=fx['target_date'])
                math_results.append(res)
                
        except Exception as e:
            math_results.append(f"Quantitative Parsing Error: {str(e)}")
            
        return {
            "math_expressions": expressions_used,
            "math_results": math_results,
            "messages": [("system", f"Quantitative Agent executed {len(math_results)} tool operations.")]
        }