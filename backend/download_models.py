import os
import urllib.request
from huggingface_hub import snapshot_download

def download_models():
    """
    Downloads the required baseline and LLM model files from Hugging Face Hub.
    In an industry setting, you would replace 'YOUR_HF_USERNAME/nhs_mental_health_triage_ai'
    with your actual Hugging Face model repository.
    """
    # Create the target directory
    models_dir = os.path.join(os.path.dirname(__file__), "models_saved")
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Download Baseline Model (Joblib)
    baseline_path = os.path.join(models_dir, "baseline_pipeline.joblib")
    if not os.path.exists(baseline_path):
        print("Downloading Baseline model from Hugging Face Hub...")
        # URL for raw file download from Hugging Face
        url = "https://huggingface.co/moseyda/nhs-triage-prototype/resolve/main/baseline_pipeline.joblib"
        urllib.request.urlretrieve(url, baseline_path)
        print("  -> Baseline model downloaded.")
    else:
        print("Baseline model already exists.")

    # 2. Download Fine-Tuned LLM (Directory of weights)
    llm_dir = os.path.join(models_dir, "llm_finetuned")
    if not os.path.exists(llm_dir):
        print("Downloading Fine-tuned LLM from Hugging Face Hub...")
        repo_id = "moseyda/nhs-triage-prototype"
        
        # We only want to download the llm_finetuned subfolder from the repo
        snapshot_download(
            repo_id=repo_id, 
            local_dir=models_dir, 
            allow_patterns=["llm_finetuned/*"]
        )
        print("  -> LLM downloaded.")
    else:
        print("Fine-tuned LLM already exists.")

if __name__ == "__main__":
    download_models()
    print("\nModel setup complete! You can now run the API.")
