import os
import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nlp.data_utils import load_synthetic_referral_data, preprocess_data, get_train_val_test_splits
from nlp.train_baseline import predict_risk_baseline
from nlp.prioritisation import get_priority_band, calculate_prioritisation_score
from nlp.evaluation import evaluate_triage_queue
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def generate_report():
    print("Loading test data...")
    df = load_synthetic_referral_data(num_samples=1000)
    df = preprocess_data(df)
    _, _, test_df = get_train_val_test_splits(df)
    
    # 1. Evaluate Baseline Model
    print("Evaluating Baseline Model (TF-IDF + LR)...")
    base_results = []
    
    for idx, row in test_df.iterrows():
        out = predict_risk_baseline(row['text'])
        risk_prob = out['risk_probabilities'][2]
        band = get_priority_band(risk_prob)
        score = calculate_prioritisation_score(risk_prob, band)
        
        base_results.append({
            'text': row['text'],
            'true_label': row['label'],
            'predicted_class': out['predicted_class'],
            'high_risk_prob': risk_prob,
            'prioritisation_score': score
        })
    
    baseline_df = pd.DataFrame(base_results)
    
    # 2. Evaluate LLM
    print("Evaluating Fine-tuned LLM...")
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models_saved", "llm_finetuned")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    llm_results = []
    
    for idx, row in test_df.iterrows():
        encoding = tokenizer(
            str(row['text']).lower(),
            return_tensors='pt',
            max_length=128,
            padding='max_length',
            truncation=True
        )
        with torch.no_grad():
            outputs = model(**encoding)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1).squeeze().tolist()
            pred_class = torch.argmax(logits, dim=-1).item()
            
        risk_prob = probs[2]
        band = get_priority_band(risk_prob)
        score = calculate_prioritisation_score(risk_prob, band)
        
        llm_results.append({
            'text': row['text'],
            'true_label': row['label'],
            'predicted_class': pred_class,
            'high_risk_prob': risk_prob,
            'prioritisation_score': score
        })

    llm_df = pd.DataFrame(llm_results)

    # 3. Calculate Final Metrics
    print("Calculating metrics...")
    
    def calc_standard_metrics(result_df):
        acc = accuracy_score(result_df['true_label'], result_df['predicted_class'])
        p, r, f1, _ = precision_recall_fscore_support(
            result_df['true_label'], result_df['predicted_class'], average='macro'
        )
        return {"accuracy": acc, "precision": p, "recall": r, "f1_score": f1}

    base_metrics = calc_standard_metrics(baseline_df)
    base_triage = evaluate_triage_queue(baseline_df, 'true_label', 'prioritisation_score')
    
    llm_metrics = calc_standard_metrics(llm_df)
    llm_triage = evaluate_triage_queue(llm_df, 'true_label', 'prioritisation_score')

    report = {
        "baseline_model": {
            "classification_metrics": base_metrics,
            "triage_queue_metrics": base_triage
        },
        "fine_tuned_llm": {
            "classification_metrics": llm_metrics,
            "triage_queue_metrics": llm_triage
        }
    }

    # 4. Save to Disk
    out_dir = os.path.join(os.path.dirname(__file__), "..", "nlp", "logs")
    os.makedirs(out_dir, exist_ok=True)
    
    baseline_df.to_csv(os.path.join(out_dir, "results_baseline.csv"), index=False)
    llm_df.to_csv(os.path.join(out_dir, "results_llm.csv"), index=False)
    
    with open(os.path.join(out_dir, "evaluation_report.json"), "w") as f:
        json.dump(report, f, indent=4)
        
    # Also save a human-readable text file
    with open(os.path.join(out_dir, "evaluation_report.txt"), "w") as f:
        f.write("DISSERTATION EVALUATION REPORT\n")
        f.write("==============================\n\n")
        for model_name, data in report.items():
            f.write(f"--- {model_name.upper()} ---\n")
            f.write("Classification Metrics:\n")
            for k, v in data["classification_metrics"].items():
                f.write(f"  {k}: {v:.4f}\n")
            f.write("Triage Metrics:\n")
            for k, v in data["triage_queue_metrics"].items():
                 f.write(f"  {k}: {v:.4f}\n")
            f.write("\n")

    print(f"Done! Evidence saved successfully to {out_dir}")

if __name__ == "__main__":
    generate_report()
