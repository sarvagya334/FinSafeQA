import json
import re
from typing import Dict, Any
from langchain_core.messages import AIMessage
from agents.state import AgentState 
# Ensure these imports match your actual tool file structure
from mcp_tools.math_repl import finance_calculator
from mcp_tools.triangular_fx_tool import fx_tool

class QuantitativeAgent:
    """
    An agent that performs deterministic financial calculations and FX conversions
    by interpreting the current conversation and regional policies.
    """
    def __init__(self, llm_callable):
        self.llm = llm_callable

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        # 1. Gather the full context of the "Meeting"
        chat_transcript = "\n".join(
            [f"{msg.name or msg.type}: {msg.content}" for msg in state["messages"]]
        )
        
        prompt = f"""
        You are the Quantitative Specialist in a financial multi-agent system.
        Review the conversation history and the policies retrieved by the regional agents.
        
        --- MEETING TRANSCRIPT ---
        {chat_transcript}
        --------------------------
        
        Your task:
        1. Extract the interest rates, tax percentages, and investment amounts mentioned.
        2. Generate a JSON object to run the 'math_expression' or an 'fx_request'.
        
        Constraints:
        - "math_expression": Use Python math syntax (e.g., 1000 * (1.08 ** 5)).
        - "fx_request": Provide "amount" (in SGD) and "target_date" (YYYY-MM-DD).
        
        Respond ONLY with a JSON object.
        """
        
        raw_response = self.llm(prompt)
        math_results = []
        
        try:
            # Clean and parse the LLM's instruction
            clean_json = re.sub(r"```json|```", "", raw_response).strip()
            instructions = json.loads(clean_json)
            
            # Execute Math Tool (Deterministic calculation)
            if "math_expression" in instructions:
                expr = instructions["math_expression"]
                res = finance_calculator.execute(expr)
                math_results.append(f"Math Result ({expr}): {res}")
                
            # Execute FX Tool (Historical currency lookup)
            if "fx_request" in instructions:
                fx = instructions["fx_request"]
                res = fx_tool.convert_sgd_to_inr(amount=fx['amount'], target_date=fx['target_date'])
                math_results.append(f"FX Result ({fx['amount']} SGD on {fx['target_date']}): {res}")
                
        except Exception as e:
            math_results.append(f"Calculation Error: {str(e)}")
            
        # 3. Create the conversational reply for the group
        if math_results:
            reply_content = (
                "I have processed the quantitative data based on the policies shared above:\n\n" + 
                "\n".join([f"- {r}" for r in math_results]) + 
                "\n\n@Synthesizer, please incorporate these figures into the final comparison."
            )
        else:
            reply_content = "I reviewed the context but didn't find specific variables for calculation. @Synthesizer, proceed with a qualitative summary."

        # 4. Update the state and route to the next agent
        return {
            "messages": [AIMessage(content=reply_content, name="Quant_Engine")],
            "next_speaker": "Synthesizer" 
        }