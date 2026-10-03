from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    #Ollama
    OLLAMA_BASE_URL: str= "http://localhost:11434"
    OLLAMA_GENERATION_MODEL: str= "qwen3:8b"
    OLLAMA_EMBEDDING_MODEL: str= "nomic-embed-text"

    #Pinecone
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str= "college-notes"

    #RAG
    TOP_K: int= 5
    CHUNK_SIZE: int= 500
    CHUNK_OVERLAP: int= 50

    model_config = SettingsConfigDict(
        env_file= ".env",
        env_file_encoding= "utf-8",
        case_sensitive= False
    )

settings = Settings()