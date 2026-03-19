# backend/app/api.py
from fastapi import APIRouter, HTTPException
from .schemas import ReferralRequest, TriageResponse
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
            prioritisation_score=result["prioritisation_score"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
