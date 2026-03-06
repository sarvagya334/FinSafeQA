import duckdb
import os
import glob

class SQLQueryTool:
    """
    An MCP tool that executes read-only SQL queries directly against 
    the raw financial and macroeconomic CSV datasets.
    """
    def __init__(self, data_dir: str):
        # Initialize an in-memory DuckDB instance
        self.conn = duckdb.connect(database=':memory:')
        
        # Define paths to the three core data categories
        # Note: glob finds all files matching the *.csv pattern
        paths = {
            "india": os.path.join(data_dir, "raw", "india_stable_assets", "*.csv"),
            "sg": os.path.join(data_dir, "raw", "singapore_stable_assets", "*.csv"),
            "macro": os.path.join(data_dir, "raw", "macro_and_fx", "*.csv")
        }
        
        # Register each directory
        for prefix, path_pattern in paths.items():
            self._register_views(prefix, path_pattern)

    def _register_views(self, prefix: str, path_pattern: str):
        """Creates virtual tables (views) for every CSV found."""
        for file in glob.glob(path_pattern):
            # Extract a clean table name from the filename
            base_name = os.path.basename(file).replace(".csv", "").replace(" ", "_").replace("-", "_").lower()
            table_name = f"{prefix}_{base_name}"
            
            # CHANGE '{file_path}' TO '{file}' BELOW
            query = f"""
            CREATE VIEW '{table_name}' AS 
            SELECT * FROM read_csv_auto(
                '{file}', 
                ignore_errors=true, 
                null_padding=true,
                all_varchar=true,
                strict_mode=false
            );
            """
            
            try:
                self.conn.execute(query)
            except Exception as e:
                print(f"Warning: Could not register view for {file} - {str(e)}")

    def execute_query(self, sql_string: str) -> str:
        """
        Executes an LLM-generated SELECT query and returns the results.
        """
        # Security Guardrail: Prevent destructive commands in the OOPD environment
        forbidden = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE"]
        if any(keyword in sql_string.upper() for keyword in forbidden):
            return "SQL Error: Only SELECT queries are permitted for financial safety."
            
        try:
            # Execute and convert directly to a Pandas DataFrame for easy formatting
            result_df = self.conn.execute(sql_string).df()
            
            if result_df.empty:
                return "SQL Result: Query executed successfully but returned 0 rows."
            
            # Limiting to 10 rows keeps the LLM context window from overflowing
            return f"SQL Result (Top 10 rows):\n{result_df.head(10).to_string()}"
            
        except Exception as e:
            return f"SQL Execution Error: {str(e)}"

# Instantiate for global access across the agent nodes
sql_tool = SQLQueryTool(data_dir="data")