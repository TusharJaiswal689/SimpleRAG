import json

from fastapi import APIRouter, Depends
from fastapi import Request as HTTPRequest
from fastapi.responses import StreamingResponse

from app.api.schemas import ChatRequest
from app.core.dependencies import get_rag_service


router = APIRouter(
    prefix="/chat",
    tags=["Chat"]
)


@router.post("/")
def chat(
    request: ChatRequest,
    http_request: HTTPRequest,
    rag_service=Depends(get_rag_service)
):
    history = [message.model_dump() for message in request.history]
    if "text/event-stream" in http_request.headers.get("accept", ""):
        def encode_events():
            for event in rag_service.generate_events(request.query, history):
                data = json.dumps(event["data"], ensure_ascii=False)
                yield f"event: {event['event']}\ndata: {data}\n\n"

        return StreamingResponse(
            encode_events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return StreamingResponse(
        rag_service.generate_answer(request.query, history),
        media_type="text/plain"
    )