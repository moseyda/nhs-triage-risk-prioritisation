import pandas as pd
from .data_utils import load_synthetic_referral_data, preprocess_data, get_train_val_test_splits
from .train_baseline import predict_risk_baseline
from .prioritisation import get_priority_band, calculate_prioritisation_score
from .evaluation import evaluate_triage_queue

def run_triage_simulation():
    """
    Simulates a triage queue by running test referrals through the complete pipeline
    and evaluating how well the pipeline surfaces high-risk cases.
    """
    print("Loading test data for simulation...")
    df = load_synthetic_referral_data(num_samples=1000)
    df = preprocess_data(df)
    _, _, test_df = get_train_val_test_splits(df)
    
    results = []
    
    print("Running baseline model on test cases to generate triage queue...")
    for idx, row in test_df.iterrows():
        # Get raw prediction
        prediction_output = predict_risk_baseline(row['text'])
        
        # High risk is class 2; we use its probability as the risk score
        risk_prob = prediction_output['risk_probabilities'][2]
        
        # Apply prioritisation rules
        band = get_priority_band(risk_prob)
        score = calculate_prioritisation_score(risk_prob, band)
        
        results.append({
            'text': row['text'],
            'label': row['label'],
            'predicted_class': prediction_output['predicted_class'],
            'risk_prob_high': risk_prob,
            'band': band,
            'prioritisation_score': score
        })
        
    results_df = pd.DataFrame(results)
    
    print("\nEvaluating Triage Queue performance...")
    metrics = evaluate_triage_queue(results_df, true_label_col='label', score_col='prioritisation_score')
    
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        
    print("\nTop 5 Cases in Triage Queue:")
    sorted_queue = results_df.sort_values(by='prioritisation_score', ascending=False)
    print(sorted_queue[['text', 'label', 'band', 'prioritisation_score']].head(5))

if __name__ == "__main__":
    run_triage_simulation()
