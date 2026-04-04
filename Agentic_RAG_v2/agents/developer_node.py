import json
import re
from typing import Dict, Any
from langchain_core.messages import AIMessage
# Use the unified AgentState
from agents.state import AgentState 
from mcp_tools.trend_plotter import plot_tool

class DeveloperAgent:
    """
    An autonomous agent that writes and executes Python code for data visualization.
    Communicates with the team via the shared message state.
    """
    def __init__(self, llm_callable):
        self.llm = llm_callable

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        # 1. Pull the user's intent from the message history
        user_query = state["messages"][0].content
        chat_transcript = "\n".join(
            [f"{msg.name or msg.type}: {msg.content}" for msg in state["messages"]]
        )
        
        prompt = f"""
        You are the Developer Agent for a financial analysis team.
        Review the conversation and user request below:
        
        --- CONVERSATION ---
        {chat_transcript}
        --------------------

        Task: Write a Python script to generate a chart using `pd` (pandas) and `plt` (matplotlib).
        Data Path: Use the directory `DATA_DIR`.
        
        Available Tables (Query via duckdb):
        - `india_singapore_data - Long_Term_Govt_Bond_Yield_India.csv`
        - `india_singapore_data - SORA.csv`
        - `india_singapore_data - CPI_India.csv`
        
        Output ONLY a strict JSON object with a "python_script" key. 
        Do NOT include plt.show().

        Example:
        {{
            "python_script": "df = duckdb.query(\\"SELECT observation_date, SORA FROM read_csv_auto(f'{{DATA_DIR}}/india_singapore_data - SORA.csv')\\").df()\\nplt.plot(df['observation_date'], df['SORA'])"
        }}
        """
        
        raw_response = self.llm(prompt)
        
        try:
            clean_json = re.sub(r"```json|```", "", raw_response).strip()
            instructions = json.loads(clean_json)
            
            if "python_script" in instructions:
                script = instructions["python_script"]
                # Execute the code
                plot_status = plot_tool.execute_plotting_script(script)
                
                # 2. Formulate a message to the group chat
                reply_content = (
                    f"I have successfully executed the following plotting script:\n\n"
                    f"```python\n{script}\n```\n"
                    f"Status: {plot_status}. The chart has been saved to the buffer. "
                    f"@Synthesizer, the visual data is ready for the final report."
                )
                
                return {
                    "messages": [AIMessage(content=reply_content, name="Developer")],
                    "next_speaker": "Synthesizer" # Hand off to the writer
                }
                
        except Exception as e:
            error_msg = f"Developer Agent encountered an error: {str(e)}. @Orchestrator, I may need a different data source."
            return {
                "messages": [AIMessage(content=error_msg, name="Developer")],
                "next_speaker": "Orchestrator" # Ask for help if code fails
            }