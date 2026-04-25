# backend/app/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "NHS Mental Health Triage CDSS Prototype"
    API_V1_STR: str = "/api/v1"
    
    # Priority bands defined by the system design
    BAND_HIGH: str = "High"
    BAND_MEDIUM: str = "Medium"
    BAND_LOW: str = "Low"

settings = Settings()
