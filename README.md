# NHS Mental Health Triage AI Prototype

This repository contains the prototype code for the final year Computer Science dissertation: *"Enhancing Transformer-Based NLP (BERT/LLM) for Self-Harm Risk Prioritisation in Simulated NHS Mental Health Triage"* by Mo Seyda.

## Overview
This project simulates an NHS triage environment where referral-style free-text is analysed by an AI to estimate clinical risk. The goal is to safely support, rather than replace, clinician-led prioritisation.

The system features:
1. **Baseline Model:** A TF-IDF + Logistic Regression pipeline.
2. **Enhanced NLP Pipeline:** A Hugging Face BERT (`bert-base-uncased`) fine-tuned to predict self-harm risk from unstructured text.
3. **Simulated Triage Logic:** (In Progress) Converts probability distributions into clinical priority bands and sorting queues.
4. **FastAPI Backend:** A scalable REST interface designed to decouple model inference from the frontend.
5. **React Frontend:** (Upcoming) A clinician user interface to review cases and model recommendations.

## Project Architecture
The project is strictly modularized to separate backend logic, NLP ML pipelines, and the React clinician frontend:

```text
nhs_mental_health_triage_ai/
├── backend/            # FastAPI integration
│   ├── app/            # REST API Routes, schemas, and config
│   ├── nlp/            # Model training and data preprocessing
│   ├── models_saved/   # Excluded from Git - generated locally or downloaded via HF Hub
├── frontend/           # (Upcoming) Clinician SPA Dashboard
└── PROGRESS.md         # Detailed checklist of project development phases
```

## Running the Project

**1. Clone the repository:**
```bash
git clone https://github.com/moseyda/nhs_mental_health_triage_ai.git
cd nhs_mental_health_triage_ai
```

**2. Setup Python Virtual Environment:**
```bash
cd backend
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
# source venv/bin/activate

pip install -r requirements.txt
```

**3. Download the ML Models:**
The trained `.joblib` and HuggingFace chunks (`~3GB`) are not stored in this GitHub repository. Run the provided script to pull them from the Hugging Face Hub registry:
```bash
# Requires you to install tokenizers, sentencepiece, evaluate and accelerate manually if missing
# See PROGRESS.md for more info.
python download_models.py
```

**4. Start the FastAPI Server:**
```bash
python -m app.main
```
The API will be available at `http://localhost:8000/docs`, where you can test the `/api/v1/predict` endpoint.

## Development Status
* **Phase 1-4 Complete:** Architecture agreed, FastAPI backend stubbed, Baseline trained, LLM fine-tuned.
* **Phase 5-7 Pending:** Probability calibration, prioritisation sorting logic, API integration, and the React Clinician Frontend.
