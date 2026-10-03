from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes_chat import router as chat_router
from app.api.routes_documents import router as documents_router


app = FastAPI(
    title="College Notes RAG API",
    version="1.0.0"
)


app.include_router(chat_router)
app.include_router(documents_router)


frontend_dir = Path("frontend")

app.mount(
    "/static",
    StaticFiles(directory=frontend_dir),
    name="static"
)


@app.get("/")
def root():
    return FileResponse(
        frontend_dir / "index.html"
    )