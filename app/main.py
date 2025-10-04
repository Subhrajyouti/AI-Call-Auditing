from fastapi import FastAPI
from app.routes import call_upload, transcription, scoring

app = FastAPI(title="AI Call Auditing Backend")

# Include routes
app.include_router(call_upload.router)
app.include_router(transcription.router)
app.include_router(scoring.router)

@app.get("/")
def root():
    return {"message": "AI Call Auditing API is running"}
