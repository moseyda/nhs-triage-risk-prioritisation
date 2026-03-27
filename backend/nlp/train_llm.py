import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from .data_utils import load_synthetic_referral_data, preprocess_data, get_train_val_test_splits

# We use 'bert-base-uncased' as a standard baseline encoder LLM for the prototype
# It is widely supported by HuggingFace Tokenizers
MODEL_NAME = "bert-base-uncased" 
NUM_LABELS = 3 # 0: Low, 1: Medium, 2: High

class ReferralDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(eval_pred):
    """
    Computes classification metrics for the Hugging Face Trainer.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    acc = accuracy_score(labels, predictions)
    
    # Optional: Calculate ROC AUC if logits can be safely converted to probabilities
    try:
        # Softmax to get probabilities
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        auc = roc_auc_score(labels, probs, multi_class='ovr')
    except Exception:
        auc = 0.0

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc
    }

def train_and_evaluate_llm():
    print(f"Loading data to fine-tune {MODEL_NAME}...")
    df = load_synthetic_referral_data(num_samples=5000)
    df = preprocess_data(df)
    
    train_df, val_df, test_df = get_train_val_test_splits(df)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Reset indices for smooth Dataset iteration
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    train_dataset = ReferralDataset(train_df['clean_text'], train_df['label'], tokenizer)
    val_dataset = ReferralDataset(val_df['clean_text'], val_df['label'], tokenizer)
    test_dataset = ReferralDataset(test_df['clean_text'], test_df['label'], tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    
    save_dir = os.path.join(os.path.dirname(__file__), "..", "models_saved", "llm_finetuned")
    
    # Define training arguments (optimized for CPU/Prototype testing)
    training_args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=3,              # Keep low for prototype
        per_device_train_batch_size=16,  # Math corrected for Gradient updates (down from 64)
        per_device_eval_batch_size=16,   
        warmup_steps=65,                 # Calculated ~10% of total steps for proper convergence
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",      # Evaluate at the end of each epoch
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    print("Starting LLM fine-tuning...")
    trainer.train()

    print("Evaluating best model on test set...")
    results = trainer.evaluate(test_dataset)
    print(f"Test Results: {results}")

    print(f"Saving finalized model and tokenizer to {save_dir}...")
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    print("Done fine-tuning LLM.")

def predict_risk_llm(text: str) -> dict:
    """
    Inference interface to be loaded by the API main models.py in the future.
    """
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models_saved", "llm_finetuned")
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Fine-tuned model not found at {model_dir}. Need to run train_llm.py first.")
        
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Ensure model is in eval mode
    model.eval()
    
    encoding = tokenizer(
        text.lower(),
        return_tensors='pt',
        max_length=128,
        padding='max_length',
        truncation=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1).squeeze().tolist()
        prediction = torch.argmax(logits, dim=-1).item()
        
    bands = {0: "Low", 1: "Medium", 2: "High"}
    
    return {
        "risk_probabilities": probabilities,
        "predicted_class": int(prediction),
        "suggested_band": bands[int(prediction)]
    }

if __name__ == "__main__":
    train_and_evaluate_llm()
