from fastapi import APIRouter, UploadFile
import shutil
import os

router = APIRouter(prefix="/calls", tags=["Calls"])

UPLOAD_DIR = "data/"

@router.post("/upload")
async def upload_call(file: UploadFile):
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"message": "File uploaded", "path": filepath}
