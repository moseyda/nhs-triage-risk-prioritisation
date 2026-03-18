# backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .api import router as api_router

app = FastAPI(
    title=settings.APP_NAME,
    description="Backend API for the simulated NHS mental health triage prototype.",
    version="1.0.0"
)

# Set up CORS so the React frontend can call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For the prototype, allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_STR)

if __name__ == "__main__":
    import uvicorn
    # Make sure we can run this directly string for easy testing
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
