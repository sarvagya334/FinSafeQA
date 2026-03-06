import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import duckdb
import os

class TrendPlotterTool:
    """
    A sandboxed execution environment that allows the Developer Agent to run 
    pandas and matplotlib scripts for generating financial charts.
    """
    def __init__(self, data_dir: str):
        # We pass the same DuckDB connection logic so the plotting 
        # script can query the exact same CSV data.
        self.db_path = os.path.join(data_dir, "raw", "macro_and_fx")

    def execute_plotting_script(self, script: str) -> str:
        """
        Executes a Python plotting script and returns the chart as a Base64 string.
        """
        # 1. Security check: Prevent unauthorized library imports
        forbidden = ["os", "sys", "subprocess", "shutil", "requests", "socket"]
        if any(f"import {lib}" in script or f"from {lib}" in script for lib in forbidden):
            return "Plotting Error: Unauthorized system library import detected."

        # 2. Setup the sandbox environment
        # Only explicitly allow pandas, pyplot, and a fresh duckdb connection
        local_env = {
            "pd": pd,
            "plt": plt,
            "duckdb": duckdb.connect(database=':memory:'),
            "DATA_DIR": self.db_path
        }

        # 3. Force the script to save to an in-memory buffer rather than disk
        prefix = """
import io
import base64
plt.switch_backend('Agg') # Ensure no GUI windows open
plt.figure(figsize=(10, 6))
"""
        suffix = """
buf = io.BytesIO()
plt.savefig(buf, format='png', bbox_inches='tight')
buf.seek(0)
base64_img = base64.b64encode(buf.read()).decode('utf-8')
plt.close()
"""
        executable_code = prefix + script + suffix

        try:
            # Execute the isolated code
            exec(executable_code, {}, local_env)
            
            # Retrieve the base64 string from the local environment
            b64_output = local_env.get("base64_img")
            if b64_output:
                return f"Chart successfully generated. Base64 Data: data:image/png;base64,{b64_output[:50]}...[TRUNCATED]"
            else:
                return "Plotting Error: No image data was generated in the buffer."
                
        except Exception as e:
            return f"Plotting Execution Error: {str(e)}"

# Instantiate globally
plot_tool = TrendPlotterTool(data_dir="data")