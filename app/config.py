from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    llm_provider: str ="openai"

    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    groq_api_key: str | None = None 

    embedding_provider: str = "openai"
    vector_db: str = "chromadb"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

settings = Settings()