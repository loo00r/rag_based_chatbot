from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")

    LLM_BASE_URL: str = "http://localhost:8001/v1"
    HF_TOKEN: str = ""

    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    POSTGRES_DB: str

    @computed_field
    @property
    def PG_CONN(self) -> str:
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )


_s = Settings()
LLM_BASE_URL = _s.LLM_BASE_URL
PG_CONN = _s.PG_CONN
HF_TOKEN = _s.HF_TOKEN

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION = "docs"
TOP_K = 5
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
