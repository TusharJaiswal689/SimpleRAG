from fastapi import FastAPI

from app.api.routes_chat import router as chat_router
from app.api.routes_documents import router as documents_router


app = FastAPI(
    title="College Notes RAG API",
    version="1.0.0"
)


app.include_router(chat_router)
app.include_router(documents_router)


@app.get("/")
def root():
    return {
        "message": "College Notes RAG API is running"
    }