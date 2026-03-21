import os
import pandas as pd
import torch
import shutil
import datetime
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# We assume the model was compiled and saved to models_saved/llm
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models_saved", "llm")

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
        encoding = self.tokenizer.encode_plus(
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
        
    df = pd.read_csv(csv_file)
    if len(df) == 0:
        return False, "Feedback file is empty."
        
    # Map text labels to integers mathematically
    label_map = {"Low": 0, "Medium": 1, "High": 2}
    df['label'] = df['human_corrected_band'].map(lambda x: label_map.get(str(x).capitalize(), 0))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[MLOps] Active Learning Retraining initiated on {device} with {len(df)} ground-truth override records.")
    
    if not os.path.exists(MODEL_DIR):
        return False, "Source LLM weights not found. Cannot retrain."

    # Load existing model and tokenizer safely into VRAM
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=3)
    model = model.to(device)
    model.train()
    
    dataset = FeedbackDataset(df['referral_text'].values, df['label'].values, tokenizer)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
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
    
    # Save the mathematically updated model back to disk
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    
    # Archive the CSV so the model doesn't endlessly train on the exact same records
    archive_name = os.path.join(os.path.dirname(__file__), "..", f"feedback_loop_archived_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    shutil.move(csv_file, archive_name)
    
    return True, f"Successfully fine-tuned model on {len(df)} cases and archived feedback data."

if __name__ == "__main__":
    success, msg = run_active_learning()
    print(msg)
