from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")

    LLM_BASE_URL: str
    HF_TOKEN: str

    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    POSTGRES_DB: str

    EMBED_MODEL: str
    COLLECTION: str
    TOP_K: int
    CHUNK_SIZE: int
    CHUNK_OVERLAP: int


settings = Settings()

PG_CONN = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
LLM_BASE_URL = settings.LLM_BASE_URL
HF_TOKEN = settings.HF_TOKEN
EMBED_MODEL = settings.EMBED_MODEL
COLLECTION = settings.COLLECTION
TOP_K = settings.TOP_K
CHUNK_SIZE = settings.CHUNK_SIZE
CHUNK_OVERLAP = settings.CHUNK_OVERLAP
