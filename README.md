# NHS Mental Health Triage BERT-Based CDSS Prototype

This repository contains the clinical decision support prototype for the Computer Science dissertation: *"Enhancing Transformer-Based NLP (BERT/LLM) for Self-Harm Risk Prioritisation in Simulated NHS Mental Health Triage"* by Mo Seyda.

## Disclaimer
This prototype is for research and educational purposes only. It is not a medical device and must not be used for real clinical decision-making. All data used are synthetic.

## Problem Statement & Engineering Intent (Spec)

**The Problem:** In mission-critical healthcare environments like NHS mental health triage, high referral volumes create dangerous bottlenecks. Purely automated AI is unsafe for clinical decisions, while purely manual triage is too slow. 

**The Intent:** Engineer a "Human-in-the-Loop" Clinical Decision Support System (CDSS) that prioritizes patient safety and architectural scalability. The engineering spec requires:
1. **Strict Decoupling:** A clear separation of concerns between the ML inference engine (**Python/FastAPI**) and the clinician dashboard (**TypeScript/React**).
2. **Mission-Critical Reliability:** Implementing Explainable AI (XAI) so clinicians can trust and verify the AI's logic, ensuring the software is safe for high-stakes environments.
3. **Modern API Design:** Building robust REST endpoints that simulate real-world hospital data ingestion.

This intent-first design ensures the application extends beyond a theoretical BERT-based NLP model to reflect real-world clinical system constraints.

## Overview
This project simulates an NHS Clinical Decision Support System (CDSS) where unreviewed Electronic Health Record (EHR) referrals are automatically analysed by a transformer-based model (BERT) to estimate risk probability from referral text. The goal is to safely support human clinicians by providing a dynamically prioritised triage queue (a "Human-in-the-Loop" workflow).

The system features:
1. **Baseline NLP Model:** A TF-IDF + Logistic Regression pipeline.
2. **Enhanced LLM Pipeline:** A Hugging Face BERT (`bert-base-uncased`) Transformer fine-tuned to predict self-harm risk from unstructured, noisy clinical text.
3. **Probability-Based Triage Logic:** Converts raw probability distributions into clinical priority bands (High/Medium/Low) and continuous sorting scores (0-100).
4. **FastAPI Backend:** A scalable REST API that loads models into GPU VRAM (CUDA) on startup where available, falling back to CPU inference.
5. **React Dashboard:** A simulated NHS Electronic Health Record (EHR) interface built with Vite, React, and Lucide SVG Icons, featuring a real-time triage inbox and a side-by-side human review panel.

## Enterprise MLOps Architecture
This system is strictly decoupled into a dual-layer, production-grade Machine Learning pipeline engineered to mimic real-world NHS hospital infrastructure:

### 1. The Data Ingestion Engine (FastAPI & Swagger UI)
Located in `backend/`, this REST API simulates the hospital's central server. External General Practitioner (GP) clinics automatically transmit raw Multi-Modal JSON patient referrals (Demographics + Unstructured Symptoms) over the network directly into this API. 
* **For Evaluators:** You can manually native-test JSON data ingestion, XAI attribution matrix generation, and deterministic RAG protocol assignment directly via the auto-generated Swagger UI portal at `http://localhost:8000/docs`.

### 2. The Clinical Decision Support System (React UI)
Located in `frontend/`, this React SPA represents the Consultant Psychiatrist's bespoke dashboard. **Clinicians do not type referrals here.** Instead, they use this interface to safely review the Triage Queue of patients that the *Ingestion Engine* has already processed computationally. It allows Human-in-the-Loop psychological experts to read the clinical RAG protocols, visually interpret the Explainable AI (XAI) token-impact vectors, and submit Active Learning Ground-Truth overrides back to the PyTorch parameter space.

## Project Structure

```text
nhs-triage-risk-prioritisation/
├── backend/                 # FastAPI model inference engine
│   ├── app/                 # API entry points and backend logic
│   │   ├── api.py           # Main router (/queue, /predict)
│   │   └── services.py      # Deterministic protocol mapping and inference services
│   ├── nlp/                 # NLP models and pipelines
│   │   ├── train.py         # PyTorch fine-tuning pipeline
│   │   ├── predict.py       # Baseline TF-IDF and BERT prediction logic
│   │   └── utils.py         # Tokenisation, metrics, and explanations
│   ├── models/              # Trained model checkpoints (excluded from version control)
│   ├── data/                # Synthetic datasets and train/test splits
│   └── requirements.txt     # Backend Python dependencies
│
├── dissertation_pdf/
│  └── Mo_Seyda_Dissertation.pdf      # The final dissertation file
│
├── frontend/                # React-based clinical dashboard (Vite)
│   ├── src/
│   │   ├── components/      # Reusable UI components
│   │   │   ├── PatientList.jsx
│   │   │   ├── TriageCard.jsx
│   │   │   └── ReviewPanel.jsx
│   │   ├── pages/           # App pages
│   │   │   ├── Dashboard.jsx
│   │   │   ├── Prioritisation.jsx
│   │   │   └── Analytics.jsx
│   │   ├── hooks/           # API request hooks
│   │   │   └── useTriageApi.js
│   │   └── main.jsx         # React entry point
│   └── package.json         # Frontend dependencies
│
├── README.md                # Project documentation
└── LICENSE                  # MIT License

```


## Setup & Installation

### 1. Backend (Model Inference Engine)
Ensure you have **Python 3.11 or 3.12** installed (Python 3.14 does not guarantee pre-compiled PyTorch CUDA wheels).

```bash
git clone https://github.com/moseyda/nhs-triage-risk-prioritisation.git
cd nhs-triage-risk-prioritisation/backend

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate

# Install PyTorch with GPU Support (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Install FastAPI and ML dependencies
pip install -r requirements.txt

# Download Model Weights from Hugging Face Hub (Required)
python download_models.py

# Start the API Server
python -m app.main
```
The API spins up at `http://localhost:8000/docs`, exposing the simulated `/queue` and `/predict` endpoints.

### 2. Frontend (NHS Clinician Dashboard)
Open a new, second terminal window:
```bash
cd nhs-triage-risk-prioritisation/frontend
npm install
npm run dev
```
Navigate to `http://localhost:5173` to interact with the CDSS React prototype.

## API Example

POST /predict

```json
{
  "text": "[AGE: 45 | GENDER: M] I feel hopeless and have been thinking about ending my life."
}
```
Response (truncated):

```json
{
  "risk_score": 0.92,
  "priority_band": "High",
  "prioritisation_score": 92.0
}
```

## Evaluation & Dissertation Results
The repository includes empirical evaluation scripts that test the models on Out-Of-Distribution (OOD) realistic clinical text.
To generate the dissertation metrics:
```bash
cd backend
venv\Scripts\activate
python -m nlp.generate_report
```
*Empirical results indicate that the fine-tuned BERT model improves classification performance over the baseline, particularly in precision, when identifying high-risk self-harm indicators in referral text.*

## Limitations

- Trained on synthetic data (limited real-world generalisability)
- Sensitive to lexical cues and negation in clinical text
- Probability outputs are not fully calibrated for ranking tasks
- Not tested in a real-world NHS clinical setting
## Contributing

As this project originated as a Computer Science dissertation proof-of-concept, major architectural overhauls are not actively maintained. However, bug fixes, evaluation improvements, and minor feature additions are welcome.

If you would like to contribute:
1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/Improvement`)
3. **Ensure code follows existing patterns** (maintain strict separation between the FastAPI MLOps backend and React CDSS frontend)
4. **Test your changes** (ensure `generate_report.py` and the React UI build successfully)
5. **Submit a pull request**

## Citation

```bibtex
@software{seyda2026nhstriage,
  author = {Seyda, Mohamad},
  title = {Evaluating Transformer-Based Decision Support for NHS Mental Health Triage: High-Risk Patient Identification and Prioritisation},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/moseyda/nhs-triage-risk-prioritisation}}
}
```
