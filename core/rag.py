from langgraph.graph import StateGraph, END
from typing import TypedDict


class RAGState(TypedDict):
    query: str
    docs: list[str]
    answer: str


def retrieve(state: RAGState) -> dict:
    # TODO: query pgvector, return top-k chunks
    return {"docs": []}


def generate(state: RAGState) -> dict:
    # TODO: call llama.cpp with docs as context
    return {"answer": ""}


builder = StateGraph(RAGState)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

rag_graph = builder.compile()
