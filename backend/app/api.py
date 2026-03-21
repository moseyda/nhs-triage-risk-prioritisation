import uuid
import random
import csv
import os
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List
from .schemas import ReferralRequest, TriageResponse, PatientCase, FeedbackRequest
from .config import settings
from .services import triage_service
from nlp.retrain_active_learning import run_active_learning

router = APIRouter()

@router.get("/health", summary="Health Check")
def health_check():
    """
    Returns the health status of the API setup.
    """
    return {
        "status": "ok", 
        "message": f"{settings.APP_NAME} API is running.", 
        "models_loaded": triage_service.is_ready
    }

@router.get("/queue", response_model=List[PatientCase], summary="Get Prioritised EHR Queue")
def get_triage_queue():
    """
    Simulates fetching a batch of unreviewed EHR referrals, piping them through the AI,
    and returning them sorted by clinical priority.
    """
    mock_referrals = [
        {"age": 24, "gender": "F", "text": "Panic attacks are getting worse. Afraid to leave the house. Feel completely trapped."},
        {"age": 45, "gender": "M", "text": "i gave away my dog yesterday and cleared out my bank accounts. they will be better off without me burdening them."},
        {"age": 19, "gender": "M", "text": "My son is dealing with bad anxiety at school, how do I get him a referral?"},
        {"age": 34, "gender": "F", "text": "having dark thoughts again but i know i wont act on them because of my kids. just need my meds adjusted."},
        {"age": 60, "gender": "M", "text": "i cant take it anymore. i bought the things i need and im driving to the bridge tonight."}
    ]
    
    cases = []
    for ref in mock_referrals:
        try:
            result = triage_service.predict(ref["text"])
            triage_res = TriageResponse(
                risk_score=result["risk_score"],
                priority_band=result["priority_band"],
                prioritisation_score=result["prioritisation_score"],
                word_attributions=result.get("word_attributions", [])
            )
            cases.append(PatientCase(
                id=str(uuid.uuid4())[:8],
                mrn=f"NHS-{random.randint(100000, 999999)}",
                age=ref["age"],
                gender=ref["gender"],
                referral_text=ref["text"],
                ai_triage=triage_res
            ))
        except Exception:
            continue
            
    # Sort by Prioritisation Score descending so High Risk is forced to the top
    cases.sort(key=lambda x: x.ai_triage.prioritisation_score, reverse=True)
    return cases

@router.post("/predict", response_model=TriageResponse, summary="Predict Risk & Priority")
def predict_triage(request: ReferralRequest):
    """
    Simulated endpoint designed as if calling the final unified ML pipeline.
    """
    try:
        result = triage_service.predict(request.text)
        return TriageResponse(
            risk_score=result["risk_score"],
            priority_band=result["priority_band"],
            prioritisation_score=result["prioritisation_score"],
            word_attributions=result.get("word_attributions", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feedback", summary="Active Learning Feedback Loop")
def submit_feedback(feedback: FeedbackRequest, background_tasks: BackgroundTasks):
    """
    Captures human clinician overrides to build a retraining dataset for MLOps.
    Solves the 'Concept Drift' problem by archiving errors.
    Automatically triggers a background GPU retraining session if 5 overrides are queued.
    """
    file_path = "feedback_loop.csv"
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "patient_id", "ai_risk_score", "human_corrected_band", "referral_text"])
            
        import datetime
        writer.writerow([
            datetime.datetime.now().isoformat(),
            feedback.patient_id,
            feedback.ai_risk_score,
            feedback.human_corrected_band,
            feedback.referral_text
        ])
        
    # Check if we hit the limit to auto-retrain
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # If headers + 5 data rows = 6 lines
            if len(lines) >= 6:
                print("[MLOps] Batch threshold reached. Triggering autonomous retraining.")
                background_tasks.add_task(_background_retrain_task)
    except Exception:
        pass
        
    return {"status": "success", "message": "Feedback captured for continuous learning."}

def _background_retrain_task():
    try:
        success, result_path = run_active_learning()
        if success:
            print(f"[MLOps] Retraining complete. Hot-reloading live model weights from {result_path} into VRAM.")
            triage_service.load_models(custom_model_dir=result_path)
        else:
            print(f"[MLOps] Retraining skipped: {result_path}")
    except Exception as e:
        print(f"[MLOps] CRITICAL ERROR during background retraining: {e}")

@router.post("/trigger-retrain", summary="Force MLOps Retraining")
def force_retrain(background_tasks: BackgroundTasks):
    """
    Manual Admin trigger to mathematically unlearn shortcuts and update live weights. 
    """
    file_path = "feedback_loop.csv"
    if not os.path.isfile(file_path):
         raise HTTPException(status_code=400, detail="No feedback data to train on.")
         
    background_tasks.add_task(_background_retrain_task)
    return {"status": "ok", "message": "Active Learning retraining initiated on GPU. Live models will hot-reload in approximately 10-15 seconds."}
