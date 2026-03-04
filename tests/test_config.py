from core.config import LLM_BASE_URL, PG_CONN, EMBED_MODEL, COLLECTION, TOP_K, CHUNK_SIZE, CHUNK_OVERLAP


def test_defaults():
    assert "localhost:8001" in LLM_BASE_URL
    assert "localhost:5432" in PG_CONN
    assert TOP_K > 0
    assert CHUNK_SIZE > 0
    assert CHUNK_OVERLAP < CHUNK_SIZE


def test_types():
    assert isinstance(TOP_K, int)
    assert isinstance(CHUNK_SIZE, int)
    assert isinstance(CHUNK_OVERLAP, int)
    assert isinstance(EMBED_MODEL, str)
    assert isinstance(COLLECTION, str)
