import uuid
import random
from fastapi import APIRouter, HTTPException
from typing import List
from .schemas import ReferralRequest, TriageResponse, PatientCase
from .config import settings
from .services import triage_service

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
