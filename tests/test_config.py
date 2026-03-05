from core.config import settings, PG_CONN


def test_pg_conn_built_from_parts():
    assert "localhost" in PG_CONN or "postgres" in PG_CONN
    assert "postgresql://" in PG_CONN
    assert "@" in PG_CONN


def test_pg_conn_starts_with_scheme():
    assert PG_CONN.startswith("postgresql://")


def test_llm_base_url():
    assert settings.LLM_BASE_URL.startswith("http")
    assert "/v1" in settings.LLM_BASE_URL


def test_types():
    assert isinstance(settings.TOP_K, int)
    assert isinstance(settings.CHUNK_SIZE, int)
    assert isinstance(settings.CHUNK_OVERLAP, int)
    assert isinstance(settings.EMBED_MODEL, str)
    assert isinstance(settings.COLLECTION, str)
    assert isinstance(settings.HF_TOKEN, str)


def test_chunk_overlap_less_than_chunk_size():
    assert settings.CHUNK_OVERLAP < settings.CHUNK_SIZE


def test_top_k_positive():
    assert settings.TOP_K > 0
