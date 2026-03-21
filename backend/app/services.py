import os
import torch
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nlp.prioritisation import get_priority_band, calculate_prioritisation_score

class TriageService:
    """
    Singleton service that preloads ML models into memory once at server startup 
    and handles fast inference for API requests.
    """
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = None
        self.baseline_pipeline = None
        self.use_llm = True
        self.is_ready = False
        
    def load_models(self, custom_model_dir=None):
        print(f"Loading Triage AI Models into memory from {custom_model_dir or 'Primary Dir'}...")
        # 1. Try to load LLM
        model_dir = custom_model_dir if custom_model_dir else os.path.join(os.path.dirname(__file__), "..", "models_saved", "llm_finetuned")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(self.device)
            self.model.eval()
            self.use_llm = True
            print(f"  -> BERT LLM loaded successfully on {self.device}!")
        except Exception as e:
            print(f"  -> Could not load LLM. (Did you run download_models.py?): {e}")
            self.use_llm = False
            
        # 2. Load Baseline fallback
        baseline_path = os.path.join(os.path.dirname(__file__), "..", "models_saved", "baseline_pipeline.joblib")
        try:
            self.baseline_pipeline = joblib.load(baseline_path)
            if not self.use_llm:
                print("  -> Baseline TF-IDF loaded as fallback!")
        except Exception as e:
            pass # We only care if both fail
            
        if not self.use_llm and self.baseline_pipeline is None:
            print("  [X] CRITICAL: No models could be loaded. Model inference will fail.")
        else:
            self.is_ready = True
            
    def _compute_occlusion_attributions(self, text: str, base_risk: float) -> list:
        # Simple, fast, and robust XAI via Leave-One-Out (Occlusion) Token masking
        words = text.split()
        if len(words) > 100:
            words = words[:100] # Cap for API response speed
            
        attributions = []
        for i, word in enumerate(words):
            masked_words = words.copy()
            masked_words[i] = "[MASK]" # Replace target word with BERT's mask token
            masked_text = " ".join(masked_words)
            
            encoding = self.tokenizer(
                masked_text.lower(),
                return_tensors='pt',
                max_length=128,
                padding='max_length',
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**encoding)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze().tolist()
                masked_risk = probs[2]
                
            # If removing this word dropped the risk, the word strongly drove the HIGH risk prediction
            impact = base_risk - masked_risk
            
            attributions.append({
                "word": word,
                "impact_score": float(impact)
            })
            
        return attributions
            
    def predict(self, text: str) -> dict:
        if not self.is_ready:
            raise RuntimeError("Models are not loaded on server. Cannot process referral.")
            
        if self.use_llm:
            encoding = self.tokenizer(
                text.lower(),
                return_tensors='pt',
                max_length=128,
                padding='max_length',
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**encoding)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze().tolist()
                risk_prob = probs[2] # High risk is class 2
                
            word_attributions = self._compute_occlusion_attributions(text, risk_prob)
        else:
            # Baseline fallback (no XAI supported yet)
            probs = self.baseline_pipeline.predict_proba([text])[0]
            risk_prob = probs[2]
            
            # Dummy attributions for baseline
            word_attributions = [{"word": w, "impact_score": 0.0} for w in text.split()]
            
        # Feed exactly into Phase 5 Prioritisation Logic
        band = get_priority_band(risk_prob)
        score = calculate_prioritisation_score(risk_prob, band)
        
        return {
            "risk_score": float(risk_prob),
            "priority_band": band,
            "prioritisation_score": float(score),
            "word_attributions": word_attributions
        }

# Global singleton to be used across API calls
triage_service = TriageService()
