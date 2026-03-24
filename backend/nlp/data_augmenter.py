import google.generativeai as genai
import csv
import random
import time
import os
import json

# ==========================================
# 1. PASTE YOUR GEMINI API KEY HERE
GEMINI_API_KEY = "YOUR_API_KEY_HERE"

# 2. HOW MANY PATIENTS DO YOU WANT TO GENERATE?
TOTAL_PATIENTS_TO_GENERATE = 50 
# ==========================================

def generate_batch(batch_size=10):
    """
    Calls the Gemini API to synthetically hallucinate highly realistic NHS referrals.
    """
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    You are an expert UK NHS psychiatric triage nurse. Generate exactly {batch_size} synthetic, highly realistic mental health patient referral texts.
    They must be diverse in age (18-80), gender, and symptomatology. 
    Ensure a realistic distribution of:
    - "High" Priority (acute crisis, active suicide planning, severe psychosis requiring immediate intervention).
    - "Medium" Priority (severe impairment, urgent but safe, e.g. acute panic, severe depression without active lethal plans).
    - "Low" Priority (routine therapy, mild anxiety, bereavement).
    
    Return ONLY a raw JSON array of objects with these exact keys: "Age", "Gender", "Referral_Text", "Priority_Band". Do not include Markdown blocks.
    Example: [{{"Age": 24, "Gender": "F", "Referral_Text": "Panic attacks daily.", "Priority_Band": "Medium"}}]
    """
    
    try:
        response = model.generate_content(prompt)
        # Clean potential markdown wrapping
        raw_text = response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
            
        return json.loads(raw_text.strip())
    except Exception as e:
        print(f"Error generating batch from Gemini: {e}")
        return []

def main():
    if GEMINI_API_KEY == "YOUR_API_KEY_HERE":
        print("ERROR: Please insert your Gemini API key at the top of data_augmenter.py")
        return
        
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_triage_data.csv")
    batch_size = 10
    
    print(f"Starting Data Augmentation: Autonomously generating {TOTAL_PATIENTS_TO_GENERATE} synthetic NHS patients via Gemini AI...")
    
    # Append to the live training dataset
    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        for i in range(0, TOTAL_PATIENTS_TO_GENERATE, batch_size):
            print(f"Requesting batch {i//batch_size + 1}...")
            new_patients = generate_batch(batch_size)
            
            for p in new_patients:
                # Generate a random 6-digit mock NHS Patient ID
                patient_id = f"NHS-{random.randint(100000, 999999)}"
                writer.writerow([patient_id, p.get("Age", 30), p.get("Gender", "Unknown"), p.get("Referral_Text", "Error"), p.get("Priority_Band", "Low")])
                
            # Sleep 2 seconds to respect Gemini API free-tier rate limits
            time.sleep(2) 
            
    print(f"\nSUCCESS: Appended {TOTAL_PATIENTS_TO_GENERATE} shiny new synthetic patients to {csv_path}!")
    print("\nIMPORTANT NEXT STEP:")
    print("Because you have fundamentally changed the dataset size, you must now retrain the underlying Neural Network on this expanded dataset!")
    print("Run this command in your terminal:  python -m nlp.train_llm  ")

if __name__ == "__main__":
    main()
