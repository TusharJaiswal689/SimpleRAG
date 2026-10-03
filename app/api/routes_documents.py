from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile

from app.api.schemas import DocumentResponse
from app.core.dependencies import get_ingestion_service


router = APIRouter(
    prefix="/documents",
    tags=["Documents"]
)


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    ingestion_service=Depends(get_ingestion_service)
):
    upload_dir = Path("data/documents")
    upload_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    file_path = upload_dir / file.filename

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    return ingestion_service.ingest(
        str(file_path)
    )