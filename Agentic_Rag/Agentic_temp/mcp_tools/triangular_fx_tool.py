import pandas as pd
import os

class TriangularFXTool:
    """
    Calculates exact SGD to INR exchange rates for a given historical date
    using triangular computation via the USD base rate.
    """
    def __init__(self, data_dir: str):
        # Locate the new macro datasets
        inr_file = os.path.join(data_dir, "raw", "macro_and_fx", "india_singapore_data - INRtoUSD.csv")
        sgd_file = os.path.join(data_dir, "raw", "macro_and_fx", "india_singapore_data - singaporeDollarToUSD.csv")
        
        self.inr_df = pd.read_csv(inr_file, parse_dates=['observation_date'])
        self.sgd_df = pd.read_csv(sgd_file, parse_dates=['observation_date'])
        
        # Sort for fast nearest-date searching
        self.inr_df = self.inr_df.sort_values('observation_date').dropna()
        self.sgd_df = self.sgd_df.sort_values('observation_date').dropna()

    def convert_sgd_to_inr(self, amount: float, target_date: str) -> str:
        """
        Calculates the INR equivalent of an SGD amount on a specific date.
        """
        try:
            date_obj = pd.to_datetime(target_date)
            
            # Use merge_asof to find the closest valid trading day (handles weekends)
            inr_row = pd.merge_asof(pd.DataFrame({'observation_date': [date_obj]}), 
                                    self.inr_df, on='observation_date', direction='backward')
            sgd_row = pd.merge_asof(pd.DataFrame({'observation_date': [date_obj]}), 
                                    self.sgd_df, on='observation_date', direction='backward')
            
            inr_per_usd = float(inr_row['DEXINUS'].iloc[0])
            sgd_per_usd = float(sgd_row['DEXSIUS'].iloc[0])
            
            # Triangular Calculation: (INR / USD) / (SGD / USD) = INR / SGD
            inr_per_sgd = inr_per_usd / sgd_per_usd
            converted_amount = amount * inr_per_sgd
            
            return f"{amount} SGD on {target_date} equals {round(converted_amount, 2)} INR (Rate: 1 SGD = {round(inr_per_sgd, 4)} INR)."
            
        except Exception as e:
            return f"FX Execution Error: {str(e)}"

# Instantiate globally for the agents to import
fx_tool = TriangularFXTool(data_dir="data")