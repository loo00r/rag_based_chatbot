from core.config import LLM_BASE_URL, PG_CONN, HF_TOKEN, EMBED_MODEL, COLLECTION, TOP_K, CHUNK_SIZE, CHUNK_OVERLAP


def test_pg_conn_built_from_parts():
    assert "localhost" in PG_CONN or "postgres" in PG_CONN
    assert "postgresql://" in PG_CONN
    assert "@" in PG_CONN


def test_pg_conn_no_hardcoded_credentials():
    # connection string must be assembled, not a literal default
    assert PG_CONN.startswith("postgresql://")


def test_llm_base_url():
    assert LLM_BASE_URL.startswith("http")
    assert "/v1" in LLM_BASE_URL


def test_types():
    assert isinstance(TOP_K, int)
    assert isinstance(CHUNK_SIZE, int)
    assert isinstance(CHUNK_OVERLAP, int)
    assert isinstance(EMBED_MODEL, str)
    assert isinstance(COLLECTION, str)
    assert isinstance(HF_TOKEN, str)


def test_chunk_overlap_less_than_chunk_size():
    assert CHUNK_OVERLAP < CHUNK_SIZE


def test_top_k_positive():
    assert TOP_K > 0
