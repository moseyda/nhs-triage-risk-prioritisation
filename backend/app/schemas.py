from pydantic import BaseModel, Field
from typing import Optional, List

class ReferralRequest(BaseModel):
    text: str = Field(..., description="The free-text clinical referral to be triaged.")
    metadata: Optional[dict] = Field(default=None, description="Optional patient or referral metadata.")

class FeedbackRequest(BaseModel):
    patient_id: str = Field(..., description="The simulated EHR patient ID.")
    referral_text: str = Field(..., description="The original referral text.")
    ai_risk_score: float = Field(..., description="What the AI originally predicted.")
    human_corrected_band: str = Field(..., description="The clinician's override decision: High, Medium, or Low.")

class WordAttribution(BaseModel):
    word: str
    impact_score: float

class TriageResponse(BaseModel):
    risk_score: float = Field(..., description="The calibrated probability risk score output by the model.", ge=0.0, le=1.0)
    priority_band: str = Field(..., description="The clinical priority band: e.g., 'High', 'Medium', 'Low'.")
    prioritisation_score: float = Field(..., description="The calculated score used to rank the referral in the triage queue.")
    word_attributions: List[WordAttribution] = Field(default_factory=list, description="XAI feature attributions highlighting which words drove the risk prediction.")

class PatientCase(BaseModel):
    id: str
    mrn: str
    age: int
    gender: str
    referral_text: str
    ai_triage: TriageResponse
