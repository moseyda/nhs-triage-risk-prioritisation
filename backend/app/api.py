# backend/app/api.py
from fastapi import APIRouter
from .schemas import ReferralRequest, TriageResponse
from .config import settings

router = APIRouter()

@router.get("/health", summary="Health Check")
def health_check():
    """
    Returns the health status of the API setup.
    """
    return {"status": "ok", "message": f"{settings.APP_NAME} API is running."}

@router.post("/predict", response_model=TriageResponse, summary="Predict Risk & Priority")
def predict_triage(request: ReferralRequest):
    """
    Simulated endpoint designed as if calling the final unified ML pipeline.
    Right now it returns dummy values to fulfill the structural requirement.
    """
    
    # --- Dummy Application Logic for Output ---
    # TODO: Replace with the actual model invocation here
    # e.g., model = get_active_model() 
    #       risk_prob = model.predict(request.text)
    
    dummy_risk_score = 0.85 
    dummy_band = settings.BAND_HIGH
    dummy_prioritisation_score = 92.5
    
    return TriageResponse(
        risk_score=dummy_risk_score,
        priority_band=dummy_band,
        prioritisation_score=dummy_prioritisation_score
    )
