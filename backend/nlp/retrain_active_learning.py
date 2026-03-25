import os
import pandas as pd
import torch
import shutil
import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# We assume the model was compiled and saved to models_saved/llm_finetuned
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models_saved", "llm_finetuned")

class FeedbackDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text, add_special_tokens=True, max_length=self.max_len,
            return_token_type_ids=False, padding='max_length',
            truncation=True, return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def run_active_learning():
    csv_file = os.path.join(os.path.dirname(__file__), "..", "feedback_loop.csv")
    if not os.path.exists(csv_file):
        return False, "No feedback data to train on (feedback_loop.csv missing)."
        
    df_overrides = pd.read_csv(csv_file)
    if len(df_overrides) == 0:
        return False, "Feedback file is empty."
        
    # Map text labels to integers mathematically
    label_map = {"Low": 0, "Medium": 1, "High": 2}
    df_overrides['label'] = df_overrides['human_corrected_band'].map(lambda x: label_map.get(str(x).capitalize(), 0))
    
    # === FEATURE 1: EXPERIENCE REPLAY BUFFER ===
    # To prevent Catastrophic Forgetting of foundational anchor classes, we merge the overrides
    # with a random sample of original historical training data (Rehearsal technique).
    try:
        historical_file = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_triage_data.csv")
        df_history = pd.read_csv(historical_file)
        
        # Sample 50 records from historical training to anchor the model's memory
        sample_size = min(len(df_history), 50)
        df_history_sampled = df_history.sample(n=sample_size, random_state=42)
        df_history_sampled['label'] = df_history_sampled['Priority_Band'].map(lambda x: label_map.get(str(x).capitalize(), 0))
        
        df_hist_clean = pd.DataFrame({
            'referral_text': df_history_sampled['Referral_Text'],
            'label': df_history_sampled['label']
        })
        df_over_clean = pd.DataFrame({
            'referral_text': df_overrides['referral_text'],
            'label': df_overrides['label']
        })
        
        # Mathematically concatenate the DataFrames
        df_final = pd.concat([df_over_clean, df_hist_clean], ignore_index=True)
        print(f"[MLOps] Replay Buffer Active: {len(df_over_clean)} Overrides + {len(df_hist_clean)} Historical Anchors.")
    except Exception as e:
        df_final = pd.DataFrame({
            'referral_text': df_overrides['referral_text'],
            'label': df_overrides['label']
        })
        print(f"[MLOps] WARNING: Experience Replay Buffer failed to load. Defaulting to strict overrides: {e}")
    # ===========================================

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[MLOps] Active Learning Retraining initiated on {device} with {len(df_final)} total records.")
    
    if not os.path.exists(MODEL_DIR):
        return False, "Source LLM weights not found. Cannot retrain."

    # Load existing model and tokenizer safely into VRAM
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=3)
    model = model.to(device)
    model.train()
    
    dataset = FeedbackDataset(df_final['referral_text'].values, df_final['label'].values, tokenizer)
    loader = DataLoader(dataset, batch_size=4, shuffle=True) # Increased batch size to stabilise the mixed Replay Buffer
    
    # Very low Learning Rate (1e-5) to correct the specific override mistakes WITHOUT catastrophic forgetting of its original knowledge
    optimizer = AdamW(model.parameters(), lr=1e-5) 
    
    loss_sum = 0
    # 2 Epochs is a safe balance for tiny datasets to un-learn a shortcut correlation
    for epoch in range(2):
        for batch in loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        
    print(f"[MLOps] Concept Drift correction complete. Avg Loss: {loss_sum/(len(loader)*2)}")
    
    # Save the mathematically updated model into a NEW Rolling Checkpoint directory to prevent Windows file locks
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    new_model_dir = os.path.join(os.path.dirname(__file__), "..", "models_saved", f"llm_finetuned_{timestamp}")
    model.save_pretrained(new_model_dir)
    tokenizer.save_pretrained(new_model_dir)
    
    # Prune old checkpoints (keep the new one and the one currently locked in RAM)
    import glob
    all_checkpoints = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "models_saved", "llm_finetuned_*")))
    if len(all_checkpoints) > 3:
        for old_dir in all_checkpoints[:-3]:
            try:
                shutil.rmtree(old_dir)
            except Exception:
                pass # Ignore if still locked by OS
    
    # Archive the CSV so the model doesn't endlessly train on the exact same overrides
    archive_name = os.path.join(os.path.dirname(__file__), "..", f"feedback_loop_archived_{timestamp}.csv")
    shutil.move(csv_file, archive_name)
    
    return True, new_model_dir

if __name__ == "__main__":
    success, msg = run_active_learning()
    print(msg)
