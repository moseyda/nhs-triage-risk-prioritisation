import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from .data_utils import load_synthetic_referral_data, preprocess_data, get_train_val_test_splits

def train_and_evaluate_baseline():
    """
    Trains a simple TF-IDF + Logistic Regression model on the synthetic dataset.
    """
    print("Loading synthetic data...")
    df = load_synthetic_referral_data(num_samples=1000)
    df = preprocess_data(df)
    
    train, val, test = get_train_val_test_splits(df)
    
    print(f"Train size: {len(train)}, Val size: {len(val)}, Test size: {len(test)}")
    
    X_train, y_train = train['clean_text'], train['label']
    X_val, y_val = val['clean_text'], val['label']
    X_test, y_test = test['clean_text'], test['label']
    
    print("Building and training pipeline...")
    # Pipeline combining TF-IDF and Logistic Regression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    print("Evaluating on test set...")
    # Evaluation
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))
    
    print("--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred))
    
    try:
        # roc_auc_score requires probabilities and ovo/ovr strategy for multi-class
        auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
        print(f"\nROC AUC (One-vs-Rest): {auc:.4f}")
    except ValueError as e:
        print("ROC AUC calculation failed (class imbalance or single class present?).", str(e))
        
    # Save the model
    save_path = os.path.join(os.path.dirname(__file__), "..", "models_saved", "baseline_pipeline.joblib")
    print(f"\nSaving model pipeline to {save_path} ...")
    joblib.dump(pipeline, save_path)
    print("Done.")

def predict_risk_baseline(text: str) -> dict:
    """
    Inference interface to be loaded by the API main models.py in the future.
    Expects raw text, does the preprocessing internally and predicts using saved pipeline.
    """
    save_path = os.path.join(os.path.dirname(__file__), "..", "models_saved", "baseline_pipeline.joblib")
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Model file not found at {save_path}. Need to train first.")
        
    pipeline = joblib.load(save_path)
    clean_text = text.lower()  # Simplest preprocessing matching our data_utils
    probabilities = pipeline.predict_proba([clean_text])[0]
    prediction = pipeline.predict([clean_text])[0]
    
    # Example logic mapping labels to classes (in our synthetic data: 0: Low, 1: Medium, 2: High)
    bands = {0: "Low", 1: "Medium", 2: "High"}
    
    return {
        "risk_probabilities": probabilities.tolist(),
        "predicted_class": int(prediction),
        "suggested_band": bands[int(prediction)]
    }

if __name__ == "__main__":
    train_and_evaluate_baseline()
