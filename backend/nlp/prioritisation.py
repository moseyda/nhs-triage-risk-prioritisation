import numpy as np

def calibrate_probabilities(raw_probs: np.ndarray, method: str = "none") -> np.ndarray:
    """
    Calibrates raw model probabilities to better reflect true risk likelihood.
    
    Args:
        raw_probs: Array of shape (n_samples, n_classes) containing model probabilities.
        method: Calibration method to use (e.g., 'isotonic', 'platt', 'none').
        
    Returns:
        Calibrated probabilities.
    """
    # For the prototype MVP, we pass through the raw probabilities.
    # In a full production system, we would load a fitted IsotonicRegression 
    # or CalibratedClassifierCV model here and apply it.
    
    if method == "none":
        return raw_probs
        
    # Example placeholder for future expansion
    raise NotImplementedError(f"Calibration method '{method}' is not implemented yet.")

def get_priority_band(risk_prob: float, thresholds: dict = None) -> str:
    """
    Maps a calibrated probability to a clinical priority band.
    
    High = > 0.7
    Medium = 0.4 to 0.7
    Low = < 0.4
    """
    if thresholds is None:
        thresholds = {"high": 0.7, "medium": 0.4}
        
    if risk_prob >= thresholds["high"]:
        return "High"
    elif risk_prob >= thresholds["medium"]:
        return "Medium"
    else:
        return "Low"

def calculate_prioritisation_score(risk_prob: float, band: str, waiting_time_hours: int = 0) -> float:
    """
    Generates a sortable score (0-100) for the triage queue.
    The primary driver is the risk probability.
    We add a tiny bump for waiting time so people don't get stuck forever.
    """
    # Base score is just the probability scaled up
    base_score = risk_prob * 100.0
    
    # Add a small weight for how long they've been waiting (e.g., 0.5 points per hour)
    # This prevents medium risk patients from waiting infinitely if high risk keep coming
    time_bump = min(waiting_time_hours * 0.5, 10.0) 
    
    final_score = base_score + time_bump
    return min(final_score, 100.0) # Cap at 100
