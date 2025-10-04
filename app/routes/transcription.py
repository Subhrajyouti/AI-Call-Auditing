from fastapi import APIRouter
from app.services.asr_service import transcribe_audio

router = APIRouter(prefix="/transcription", tags=["Transcription"])

@router.get("/{filename}")
def transcribe(filename: str):
    text = transcribe_audio(f"data/{filename}")
    return {"filename": filename, "transcript": text}
