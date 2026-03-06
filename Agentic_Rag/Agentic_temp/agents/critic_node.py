from typing import Dict, Any
from agents.state import AgenticRAGState

class SynthesisAgent:
    """
    Combines the retrieved policy contexts and deterministic math results 
    into a final, highly structured, and fully cited response.
    """
    def __init__(self, llm_callable):
        self.llm = llm_callable

    def __call__(self, state: AgenticRAGState) -> Dict[str, Any]:
        prompt = f"""
        You are a Senior Financial Analyst. Synthesize a final response to the user's query.
        
        User Query: {state['query']}
        
        India Legal/Policy Context:
        {state.get('india_context', 'None')}
        
        Singapore Legal/Policy Context:
        {state.get('sg_context', 'None')}
        
        Mathematical / FX Tool Results:
        {state.get('math_results', ['No calculations performed.'])}
        
        Directives:
        1. Compare the assets logically based on the provided contexts.
        2. Use the EXACT numbers from the Mathematical Tool Results. Do not hallucinate math.
        3. Cite the "[Source X - Filename | Asset]" for every claim.
        4. Structure with headings: 'Summary', 'Comparative Analysis', 'Calculations', and 'Legal/Tax Considerations'.
        """
        
        final_answer = self.llm(prompt)
        
        return {
            "final_answer": final_answer,
            "messages": [("system", "Synthesis Agent generated the final response.")]
        }