# Project Progress Log

This document tracks the end-to-end development of the NHS Mental Health Triage AI project, from initial setup to final evaluation.

## Phase 1: Planning and Architecture
- [x] Review project requirements and `INSTRUCTIONS.md`.
- [x] Review dissertation (`Mo_Seyda_Dissertation__Copy_.pdf`) to ensure alignment with the system design and LLM scope.
- [x] Propose project structure separating backend (FastAPI/ML) and frontend (React SPA).
- [x] Refine structure to explicitly include LLM training (`train_llm.py`).
- [x] Create `PROGRESS.md` to track development.

## Phase 2: Minimal Backend Setup
- [x] Create `backend/` directory structure.
- [x] Set up minimal FastAPI app (`app/main.py`, `app/api.py`, `app/schemas.py`).
- [x] Implement `/health` and stub `/predict` endpoints.
- [x] Define initial Pydantic models for request and response.

## Phase 3: Baseline Model Pipeline
- [x] Implement data loading and preprocessing (`nlp/data_utils.py`).
- [x] Train TF-IDF + Logistic Regression model (`nlp/train_baseline.py`).
- [x] Evaluate baseline model performance.

## Phase 4: Enhanced LLM Pipeline
- [x] Setup Hugging Face Transformers and PyTorch training loop (`nlp/train_llm.py`).
- [x] Fine-tune BERT / Encoder LLM on mental health triage data.
- [x] Implement inference interface for the LLM.

## Phase 5: Calibration and Prioritisation Logic
- [x] Implement probability calibration.
- [x] Define thresholds for priority bands.
- [x] Implement logic to calculate prioritisation scores.
- [x] Implement prioritisation evaluation metrics (`nlp/evaluation.py`).

## Phase 6: API Integration
- [x] Update `/predict` to use actual ML models (Baseline or LLM).
- [x] Wire up prioritisation logic within the API response.

## Phase 7: Frontend Development (Simulated Clinician Interface)
- [x] Initialize React SPA with Vite.
- [x] Create referral submission form.
- [x] Create prioritised triage queue display.
- [x] Integrate React frontend with FastAPI backend.

## Phase 8: Deployment and Version Control
- [x] Create `.gitignore` to prevent pushing large datasets and models.
- [x] Create GitHub repository and link local Git instance.
- [x] Push committed code to remote `nhs_mental_health_triage_ai` repository.

## Phase 9: Explainable AI (XAI / Interpretability)
- [x] Implement mathematical Leave-One-Out (Occlusion) attribution in PyTorch backend.
- [x] Return token impact weights to the frontend.
- [x] Build XAI clinical insight viewer (color-coded highlighting) inside React EHR dashboard.
