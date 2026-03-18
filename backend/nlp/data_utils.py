import pandas as pd
import numpy as np
from typing import Tuple

def load_synthetic_referral_data(num_samples: int = 500) -> pd.DataFrame:
    """
    Generates a synthetic dataset of mental health referrals for prototype development.
    In a real scenario, this would load and clean an actual anonymised NHS dataset.
    """
    np.random.seed(42)
    
    # Synthetic phrases to mimic referral text
    high_risk_phrases = [
        "Patient reported active suicidal ideation with a plan.",
        "Multiple instances of severe self-harm in the past week.",
        "Overdose attempt yesterday, currently medically cleared but highly distressed.",
        "Expressing clear intent to end their life, refuses safety plan."
    ]
    
    medium_risk_phrases = [
        "Increasingly depressed, some fleeting thoughts of self-harm but no active plan.",
        "Struggling to cope with anxiety and low mood, historical self-harm.",
        "Deteriorating mental state, stopped taking medication, feels helpless.",
        "Family concerned about worsening depressive symptoms and isolation."
    ]
    
    low_risk_phrases = [
        "Seeking support for mild anxiety related to work stress.",
        "Requesting CBT for long-standing but manageable depressive symptoms.",
        "Trouble sleeping and low mood, looking for talking therapies.",
        "General stress and coping difficulties, no risk to self or others."
    ]
    
    data = []
    for _ in range(num_samples):
        # 0: Low, 1: Medium, 2: High
        risk_level = np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2])
        
        if risk_level == 2:
            text = np.random.choice(high_risk_phrases)
        elif risk_level == 1:
            text = np.random.choice(medium_risk_phrases)
        else:
            text = np.random.choice(low_risk_phrases)
            
        # Add some random noise/variation to text to make it slightly more complex for TF-IDF
        age = np.random.randint(18, 80)
        gender = np.random.choice(["M", "F", "Other"])
        noise_words = np.random.choice([" referred by GP.", " patient is anxious.", " please review.", " "], p=[0.3, 0.2, 0.2, 0.3])
        
        full_text = f"Patient ({age}{gender}): {text}{noise_words}"
        
        data.append({
            "text": full_text,
            "age": age,
            "gender": gender,
            "label": risk_level
        })
        
    return pd.DataFrame(data)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the raw data. For this prototype, we just convert text to lowercase and handle any basic cleanups.
    """
    df['clean_text'] = df['text'].str.lower()
    return df

def get_train_val_test_splits(df: pd.DataFrame, val_size=0.15, test_size=0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataframe into train, validation, and test sets.
    """
    from sklearn.model_selection import train_test_split
    
    train_val, test = train_test_split(df, test_size=test_size, random_state=42, stratify=df['label'])
    
    # Adjust val_size to be a proportion of the remaining data
    val_prop = val_size / (1.0 - test_size)
    train, val = train_test_split(train_val, test_size=val_prop, random_state=42, stratify=train_val['label'])
    
    return train, val, test
