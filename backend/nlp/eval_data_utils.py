import pandas as pd
import random

def generate_ood_evaluation_data(num_samples=500):
    """
    Generates realistic, noisy, ambiguous Out-Of-Distribution (OOD) clinical notes 
    to evaluate the model's true semantic understanding.
    """
    data = []
    
    # 0 = Low Risk, 1 = Medium Risk, 2 = High Risk (Emergency)
    for _ in range(num_samples):
        label = random.choice([0, 1, 2])
        
        if label == 0:
            templates = [
                # Hard: High risk words, low risk context (third party referral)
                "im worried about my friend, he said he might hurt himself.", 
                "My son is dealing with bad anxiety at school, how do I get him a referral?",
                # Hard: High risk words, literal physical meaning
                "i hurt my back at work and its making me depressed.",
                # Normal low risk
                "Pt reports feeling a bit down since the weather changed, requesting to restart light therapy.",
                "just need my regular sertraline script renewal plzzzz",
                "Feeling stressed about exams but managing okay with family support."
            ]
        elif label == 1:
            templates = [
                # Moderate distress, passive ideation
                "pt reports significant anhedonia, passive death wishes but denies active plan or intent.",
                "having dark thoughts again but i know i wont act on them because of my kids. just need my meds adjusted.",
                "i cnt sleep anymore. crying constantly. dont know how much longer i can keep my job like this tbh.",
                "Panic attacks are getting worse. Afraid to leave the house. Feel completely trapped in my own head.",
                "everything feels heavy and pointless. im so exhausted from pretending to be ok everyday."
            ]
        else: # label == 2
            templates = [
                # Hard: Implicit planning, no explicit trigger words (suicide/kill/die)
                "i gave away my dog yesterday and cleared out my bank accounts. they will be better off without me burdening them.", 
                "wrote the letters last night. just waiting for everyone to go to sleep now. im ready.", 
                "the pain is unbearable and i finally figured out how to make it stop permanently.",
                "Stockpiled my paroxetine for the last 3 months. Tonight is the night.",
                # Explicit action
                "i cant take it anymore. i bought the things i need and im driving to the bridge tonight."
            ]
            
        text = random.choice(templates)
        
        # Inject random clinical "noise" (typos, abbreviations)
        if random.random() < 0.3:
            text = text.replace("and", "&").replace("with", "w/").replace("about", "abt")
        if random.random() < 0.2:
             text = text.lower() # Remove all caps
             
        data.append({"text": text, "label": label})
        
    df = pd.DataFrame(data)
    # The normal pipeline expects 'clean_text', we copy 'text' to simulate raw input
    df['clean_text'] = df['text'] 
    return df
