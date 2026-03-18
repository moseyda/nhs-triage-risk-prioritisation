import numpy as np
import pandas as pd
from typing import List, Dict

def evaluate_triage_queue(df: pd.DataFrame, true_label_col: str = 'label', score_col: str = 'prioritisation_score') -> Dict[str, float]:
    """
    Calculates metrics specific to how well the queue prioritises high-risk patients.
    
    Assumes df has the true clinical label (0=Low, 1=Medium, 2=High) and 
    the model's generated prioritisation score.
    """
    # Sort the queue simulating what the clinician sees (highest score first)
    sorted_df = df.sort_values(by=score_col, ascending=False).reset_index(drop=True)
    
    total_cases = len(sorted_df)
    if total_cases == 0:
        return {}
        
    # High risk is label 2 in our synthetic dataset
    high_risk_cases = sorted_df[sorted_df[true_label_col] == 2]
    total_high_risk = len(high_risk_cases)
    
    if total_high_risk == 0:
        return {"error": "No high risk cases in the dataset to evaluate."}
        
    # 1. Mean rank of high risk cases (lower is better, 1 is the very top)
    # Ranks are 1-indexed for interpretability
    ranks = high_risk_cases.index + 1
    mean_rank = np.mean(ranks)
    
    # 2. Percentage of high risk cases caught in the top 10% of the queue
    top_10_percent_idx = max(1, int(total_cases * 0.10))
    top_10_percent_df = sorted_df.iloc[:top_10_percent_idx]
    
    high_risk_in_top_10 = len(top_10_percent_df[top_10_percent_df[true_label_col] == 2])
    recall_at_10_percent = high_risk_in_top_10 / total_high_risk
    
    # 3. Percentage of high risk cases caught in the top 20% of the queue
    top_20_percent_idx = max(1, int(total_cases * 0.20))
    top_20_percent_df = sorted_df.iloc[:top_20_percent_idx]
    
    high_risk_in_top_20 = len(top_20_percent_df[top_20_percent_df[true_label_col] == 2])
    recall_at_20_percent = high_risk_in_top_20 / total_high_risk

    return {
        "mean_rank_high_risk": mean_rank,
        "recall_at_10_percent": recall_at_10_percent,
        "recall_at_20_percent": recall_at_20_percent
    }
