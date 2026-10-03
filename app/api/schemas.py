from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str
    history: list[ChatMessage] = []

class ChatResponse(BaseModel):
    answer: str


class DocumentResponse(BaseModel):
    filename: str
    chunks_indexed: int