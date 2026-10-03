from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.schemas import ChatRequest
from app.core.dependencies import get_rag_service


router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/")
def chat(
    request: ChatRequest,
    rag_service=Depends(get_rag_service)
):
    answer_stream = rag_service.generate_answer(request.query)

    return StreamingResponse(
        answer_stream,
        media_type="text/plain"
    )