import os
from pathlib import Path
import tempfile

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

    filename = Path(file.filename or "uploaded-document").name
    target_path = upload_dir / filename
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".ingesting-",
            suffix=Path(filename).suffix,
            dir=upload_dir,
            delete=False,
        ) as buffer:
            temporary_path = Path(buffer.name)
            buffer.write(await file.read())

        result = ingestion_service.ingest(str(temporary_path), filename=filename)
        os.replace(temporary_path, target_path)
        temporary_path = None
        return result
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)