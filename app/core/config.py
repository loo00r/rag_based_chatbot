import os

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8001/v1")
PG_CONN = os.getenv("PG_CONN", "postgresql://rag:rag@localhost:5432/ragdb")

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION = "docs"
TOP_K = 5
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
