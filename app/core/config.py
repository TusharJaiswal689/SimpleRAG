from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    APP_NAME: str = "College Notes RAG"
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_GENERATION_MODEL: str = "qwen3:8b"
    OLLAMA_REWRITE_MODEL: str | None = None
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"
    OLLAMA_CONTEXT_WINDOW: int = 4096

    # Pinecone
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "college-notes"
    PINECONE_NAMESPACE: str = "phase3-test"
    PINECONE_INDEX_DIMENSION: int = 768
    PINECONE_MIN_RELEVANCE_SCORE: float = 0.70

    # Database
    DATABASE_URL: str = Field(
        validation_alias="POSTGRES_URL",
    )

    # RAG
    TOP_K: int = 5
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()