from unittest.mock import MagicMock, patch
from langgraph.graph import END
from core.agent import (
    graph, classify, decompose, rag_node, web_search, synthesize,
    route_classify, route_rag, route_synthesize,
)

BASE: dict = {
    "query": "тест", "classification": "", "sub_queries": [], "rag_answers": [],
    "web_results": [], "final_answer": "", "iterations": 0,
}
VALID_NODES = {"classify", "decompose", "rag_node", "web_search", "synthesize", "calculator_node"}


def test_graph_compiles():
    assert graph is not None


def test_graph_has_required_nodes():
    nodes = set(graph.nodes)
    for name in ["classify", "decompose", "rag_node", "web_search", "synthesize"]:
        assert name in nodes


def test_classify_returns_dict():
    assert isinstance(classify(BASE), dict)


def test_decompose_has_sub_queries():
    result = decompose(BASE)
    assert "sub_queries" in result
    assert isinstance(result["sub_queries"], list)


def test_rag_node_has_rag_answers():
    result = rag_node(BASE)
    assert "rag_answers" in result
    assert isinstance(result["rag_answers"], list)


def test_web_search_has_web_results():
    mock_result = [{"body": "Штраф за перевищення швидкості"}]
    with patch("core.agent.DDGS") as mock_ddgs:
        mock_ddgs.return_value.__enter__.return_value.text.return_value = mock_result
        result = web_search(BASE)
    assert "web_results" in result
    assert isinstance(result["web_results"], list)


def test_synthesize_increments_iterations():
    result = synthesize(BASE)
    assert result["iterations"] == 1
    assert "final_answer" in result


def test_synthesize_accumulates_iterations():
    state = {**BASE, "iterations": 1}
    result = synthesize(state)
    assert result["iterations"] == 2


def test_route_classify_returns_valid_node():
    result = route_classify(BASE)
    assert isinstance(result, str)
    assert result in VALID_NODES


def test_route_rag_returns_valid_node():
    result = route_rag(BASE)
    assert isinstance(result, str)
    assert result in VALID_NODES | {"synthesize"}


def test_route_synthesize_returns_valid():
    result = route_synthesize(BASE)
    assert result in VALID_NODES | {END}
