from typing import Dict, Any
from langchain_core.messages import AIMessage
# Assuming you updated your state.py to use the messages array
from agents.state import AgentState 

class SynthesisAgent:
    """
    Reads the entire agentic conversation and drafts the final, 
    highly structured, and fully cited response for the user.
    """
    def __init__(self, llm_callable):
        self.llm = llm_callable

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        # 1. Compile the chat history into a readable transcript for the LLM
        # This includes what the Retriever found and what the Quant calculated
        chat_transcript = "\n".join(
            [f"{msg.name or msg.type}: {msg.content}" for msg in state["messages"]]
        )
        
        prompt = f"""
        You are a Senior Financial Analyst. Synthesize a final response to the user's query.
        
        --- FULL CONVERSATION & DATA LOG ---
        {chat_transcript}
        ------------------------------------
        
        Directives:
        1. Compare the assets logically based on the policy rules provided by the Retriever in the chat.
        2. Use the EXACT numbers provided by the Quant in the chat. Do not hallucinate math.
        3. Cite the "[Source X - Filename | Asset]" for every claim.
        4. Structure with headings: 'Summary', 'Comparative Analysis', 'Calculations', and 'Legal/Tax Considerations'.
        """
        
        final_answer = self.llm(prompt)
        
        # 2. Append the final report to the messages and signal the graph to end
        return {
            "messages": [AIMessage(content=final_answer, name="Synthesizer")],
            "next_speaker": "end" # Tells the graph router that the job is done
        }