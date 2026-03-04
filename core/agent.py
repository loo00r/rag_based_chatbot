from langgraph.graph import StateGraph, END
from typing import TypedDict


class State(TypedDict):
    query: str
    sub_queries: list[str]
    rag_answers: list[str]
    web_results: list[str]
    final_answer: str
    iterations: int


def classify(state: State) -> dict:
    # TODO: LLM call → "simple" | "complex"
    return {}


def decompose(state: State) -> dict:
    # TODO: LLM call → split query into sub-questions
    return {"sub_queries": []}


def rag_node(state: State) -> dict:
    # TODO: call rag_graph for each sub_query
    return {"rag_answers": []}


def web_search(state: State) -> dict:
    # TODO: DuckDuckGo search for unanswered sub-questions
    return {"web_results": []}


def synthesize(state: State) -> dict:
    # TODO: LLM call → merge all evidence into final answer
    return {"final_answer": "", "iterations": state.get("iterations", 0) + 1}


def route_classify(state: State) -> str:
    return "decompose"  # TODO: return "rag_node" if simple


def route_rag(state: State) -> str:
    return "synthesize"  # TODO: return "web_search" if low confidence


def route_synthesize(state: State) -> str:
    return END  # TODO: loop back if incomplete and iterations < 2


builder = StateGraph(State)
builder.add_node("classify", classify)
builder.add_node("decompose", decompose)
builder.add_node("rag_node", rag_node)
builder.add_node("web_search", web_search)
builder.add_node("synthesize", synthesize)

builder.set_entry_point("classify")
builder.add_conditional_edges("classify", route_classify)
builder.add_edge("decompose", "rag_node")
builder.add_conditional_edges("rag_node", route_rag)
builder.add_edge("web_search", "synthesize")
builder.add_conditional_edges("synthesize", route_synthesize)

graph = builder.compile()
