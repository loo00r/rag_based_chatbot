from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_postgres import PGVector
from langgraph.graph import StateGraph, END
from typing import TypedDict
from core.config import PG_CONN, EMBED_MODEL, COLLECTION, TOP_K, LLM_BASE_URL

_embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
_store = PGVector(embeddings=_embeddings, collection_name=COLLECTION, connection=PG_CONN)
_llm = ChatOpenAI(base_url=LLM_BASE_URL, api_key="none", model="qwen", temperature=0)


class RAGState(TypedDict):
    query: str
    docs: list[str]
    answer: str


def retrieve(state: RAGState) -> dict:
    results = _store.similarity_search(state["query"], k=TOP_K)
    return {"docs": [f"[{d.metadata.get('rule_id','')}] {d.page_content}" for d in results]}


def generate(state: RAGState) -> dict:
    context = "\n\n".join(state["docs"])
    prompt = (
        f"Ти асистент з ПДР України. Відповідай українською, посилайся на конкретні пункти.\n\n"
        f"Контекст:\n{context}\n\nПитання: {state['query']}"
    )
    resp = _llm.invoke(prompt)
    return {"answer": resp.content}


builder = StateGraph(RAGState)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

rag_graph = builder.compile()

if __name__ == "__main__":
    result = rag_graph.invoke({"query": "Яка максимальна швидкість у населеному пункті?"})
    print(result["answer"])
