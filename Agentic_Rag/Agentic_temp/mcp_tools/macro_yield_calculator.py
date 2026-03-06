import pandas as pd
import os
from typing import Optional

class MacroYieldCalculator:
    """
    An MCP tool that adjusts nominal interest rates against historical 
    inflation data to compute the Real Yield for cross-border assets.
    """
    
    def __init__(self, data_dir: str):
        # Paths to the specific macro datasets
        india_inf_path = os.path.join(data_dir, "raw", "macro_and_fx", "india_singapore_data - InflationIndia.csv")
        sg_inf_path = os.path.join(data_dir, "raw", "macro_and_fx", "india_singapore_data - InflationSingapore.csv")
        
        # Load and prepare dataframes
        self.india_df = self._load_and_clean(india_inf_path)
        self.sg_df = self._load_and_clean(sg_inf_path)

    def _load_and_clean(self, filepath: str) -> pd.DataFrame:
        """Encapsulated helper to load and sort time-series data."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Missing macro dataset: {filepath}")
        
        df = pd.read_csv(filepath, parse_dates=['observation_date'])
        # Ensure data is sorted by date for asof merging
        return df.sort_values('observation_date').dropna(subset=['inflation'])

    def calculate_real_yield(self, nominal_rate: float, country: str, target_date: str) -> str:
        """
        Takes a nominal interest rate (e.g., 7.5 for 7.5%) and adjusts it 
        based on the inflation rate closest to the target date.
        """
        try:
            date_obj = pd.to_datetime(target_date)
            
            if country.lower() == "india":
                target_df = self.india_df
            elif country.lower() == "singapore":
                target_df = self.sg_df
            else:
                return f"Error: Unsupported country '{country}'. Must be 'India' or 'Singapore'."
            
            # Find the closest previous inflation reading
            matched_row = pd.merge_asof(
                pd.DataFrame({'observation_date': [date_obj]}), 
                target_df, 
                on='observation_date', 
                direction='backward'
            )
            
            if pd.isna(matched_row['inflation'].iloc[0]):
                return f"Warning: No inflation data found prior to {target_date} for {country}."
                
            inflation_rate = float(matched_row['inflation'].iloc[0])
            
            # The standard Fisher equation approximation: Real Yield = Nominal - Inflation
            real_yield = nominal_rate - inflation_rate
            
            return (
                f"[{country.upper()} REAL YIELD] "
                f"Nominal Rate: {nominal_rate}% | "
                f"Inflation ({matched_row['observation_date'].iloc[0].strftime('%Y-%m')}): {round(inflation_rate, 2)}% | "
                f"Real Yield: {round(real_yield, 2)}%"
            )
            
        except Exception as e:
            return f"Macro Yield Execution Error: {str(e)}"

# Instantiate globally so the Quantitative Agent can import it directly
yield_tool = MacroYieldCalculator(data_dir="data")

if __name__ == "__main__":
    # Local testing for GRS_Assingment_MT25083
    tool = MacroYieldCalculator(data_dir="../data")
    print(tool.calculate_real_yield(nominal_rate=7.5, country="India", target_date="2023-05-15"))
    print(tool.calculate_real_yield(nominal_rate=3.87, country="Singapore", target_date="2023-01-26"))