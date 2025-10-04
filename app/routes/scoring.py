# app/routes/scoring.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import csv
from app.services.asr_service import transcribe_audio
from app.services.scoring_engine import score_transcript, detect_background_noise

router = APIRouter(prefix="/audit", tags=["Scoring"])

OUTPUT_DIR = "outputs/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class AuditRequest(BaseModel):
    filename: str   # path under data/, e.g. "call123.mp3"

@router.post("/run")
def run_audit(req: AuditRequest):
    file_path = os.path.join("data", req.filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    # 1) Transcribe
    transcript = transcribe_audio(file_path)
    if not transcript or transcript.strip() == "":
        raise HTTPException(status_code=500, detail="Empty transcript returned from ASR")

    # 2) Score using scoring engine
    results = score_transcript(transcript, file_path=file_path)

    # 3) Write CSV output (columns: parameter, weight, mark, evidence)
    out_csv = os.path.join(OUTPUT_DIR, f"{os.path.splitext(req.filename)[0]}_audit.csv")
    with open(out_csv, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["parameter", "weight", "mark_obtained", "evidence"])
        for param, info in results["per_parameter"].items():
            writer.writerow([param, info["weight"], info["mark"], info["evidence"]])
        # add summary row
        writer.writerow([])
        writer.writerow(["total_score", results["total_score"]])
        writer.writerow(["max_score", results["max_score"]])

    # 4) Return JSON + csv path
    return {
        "filename": req.filename,
        "transcript_snippet": transcript[:1000],
        "scoring": results,
        "csv_path": out_csv
    }
