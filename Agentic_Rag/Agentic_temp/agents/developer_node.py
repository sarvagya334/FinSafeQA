import json
import re
from typing import Dict, Any
from agents.state import AgenticRAGState
from mcp_tools.trend_plotter import plot_tool
from mcp_tools.sql_query_tool import sql_tool

class DeveloperAgent:
    """
    An autonomous agent that writes and executes Python code for data visualization 
    using the TrendPlotterTool and SQLQueryTool.
    """
    def __init__(self, llm_callable):
        self.llm = llm_callable

    def __call__(self, state: AgenticRAGState) -> Dict[str, Any]:
        query = state["query"]
        
        prompt = f"""
        You are the Developer Agent for a Python-based financial analysis system.
        The user has requested a data visualization based on the following query: '{query}'
        
        Write a Python script using `pd` (pandas) and `plt` (matplotlib.pyplot) to generate this chart.
        You have access to a `duckdb` connection to query the CSV files. 
        Available data paths are in the directory: `DATA_DIR`.
        
        Example available tables (load via duckdb):
        - `read_csv_auto(f'{{DATA_DIR}}/india_singapore_data.xlsx - Long_Term_Govt_Bond_Yield_India.csv')`
        - `read_csv_auto(f'{{DATA_DIR}}/india_singapore_data.xlsx - SORA.csv')`
        
        Output ONLY a strict JSON object with a "python_script" key. Do NOT include `plt.show()` or `plt.savefig()`, the sandbox handles that.
        
        Example:
        {{
            "python_script": "df = duckdb.query(\\"SELECT observation_date, inflation FROM read_csv_auto(f'{{DATA_DIR}}/india_singapore_data.xlsx - InflationIndia.csv') WHERE observation_date >= '2010-01-01'\\").df()\\nplt.plot(df['observation_date'], df['inflation'])\\nplt.title('Indian Inflation Trend')"
        }}
        """
        
        raw_response = self.llm(prompt)
        
        try:
            clean_json = re.sub(r"```json|```", "", raw_response).strip()
            instructions = json.loads(clean_json)
            
            if "python_script" in instructions:
                script = instructions["python_script"]
                # Execute the code in the sandbox
                plot_result = plot_tool.execute_plotting_script(script)
                
                # Append the generated code to the state so the Critic can review it
                return {
                    "generated_code": [script],
                    "messages": [("system", f"Developer Agent executed plotting script. Status: {plot_result}")],
                    # We pass a summary to the final synthesizer
                    "math_results": [f"Visual Chart Generated: {plot_result}"] 
                }
                
        except Exception as e:
            return {"messages": [("system", f"Developer Agent Error: {str(e)}")], "math_results": []}