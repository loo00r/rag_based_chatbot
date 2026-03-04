from unittest.mock import MagicMock
import core.rag as rag_mod

BASE = {"query": "швидкість у місті", "docs": [], "answer": ""}


def test_retrieve_returns_docs():
    doc = MagicMock()
    doc.page_content = "У населеному пункті швидкість 50 км/г"
    doc.metadata = {"rule_id": "12.4"}
    rag_mod._store.similarity_search = MagicMock(return_value=[doc])

    result = rag_mod.retrieve(BASE)
    assert "docs" in result
    assert len(result["docs"]) == 1
    assert "[12.4]" in result["docs"][0]


def test_retrieve_empty():
    rag_mod._store.similarity_search = MagicMock(return_value=[])
    result = rag_mod.retrieve(BASE)
    assert result["docs"] == []


def test_generate_returns_answer():
    mock_resp = MagicMock()
    mock_resp.content = "У населеному пункті дозволено 50 км/г (п. 12.4 ПДР)"
    rag_mod._llm.invoke = MagicMock(return_value=mock_resp)

    state = {**BASE, "docs": ["[12.4] У населеному пункті..."]}
    result = rag_mod.generate(state)
    assert result["answer"] == mock_resp.content


def test_generate_passes_context_to_llm():
    rag_mod._llm.invoke = MagicMock(return_value=MagicMock(content="відповідь"))
    state = {**BASE, "docs": ["контекст1", "контекст2"]}
    rag_mod.generate(state)

    call_arg = rag_mod._llm.invoke.call_args[0][0]
    assert "контекст1" in call_arg
    assert "контекст2" in call_arg
    assert state["query"] in call_arg


def test_rag_graph_compiles():
    assert rag_mod.rag_graph is not None


def test_rag_graph_nodes():
    nodes = set(rag_mod.rag_graph.nodes)
    assert "retrieve" in nodes
    assert "generate" in nodes
