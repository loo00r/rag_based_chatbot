import json
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict
from duckduckgo_search import DDGS
from core.config import settings
from core.rag import rag_graph

_llm = ChatOpenAI(base_url=settings.LLM_BASE_URL, api_key="none", model="qwen", temperature=0)


class State(TypedDict):
    query: str
    classification: str
    sub_queries: list[str]
    rag_answers: list[str]
    web_results: list[str]
    final_answer: str
    iterations: int


def traffic_calculator(speed_kmh: int, action: str) -> str:
    if action == "stopping_distance":
        return f"Гальмівний шлях при {speed_kmh} км/г: {(speed_kmh / 10) ** 2:.0f} м"
    if action == "safe_distance":
        return f"Безпечна дистанція при {speed_kmh} км/г: {speed_kmh / 2:.0f} м"
    return "Невідома дія. Використовуй: stopping_distance або safe_distance"


# --- routing ---

def route_classify(state: State) -> str:
    routes = {
        "simple": "rag_node",
        "complex": "decompose",
        "calculation": "calculator_node",
        "out_of_scope": "web_search",
    }
    return routes.get(state.get("classification", ""), "rag_node")


def route_rag(state: State) -> str:
    return "synthesize" if state.get("rag_answers") else "web_search"


def route_synthesize(state: State) -> str:
    if state.get("iterations", 0) < 2 and not state.get("final_answer"):
        return "rag_node"
    return END


# --- nodes ---

def classify(state: State) -> dict:
    prompt = (
        "Класифікуй запит одним словом:\n"
        "- simple: пряме питання до ПДР\n"
        "- complex: порівняння або декомпозиція\n"
        "- calculation: гальмівний шлях, безпечна дистанція\n"
        "- out_of_scope: штрафи, КУпАП, поза ПДР\n\n"
        f"Запит: {state['query']}\n\nВідповідь:"
    )
    label = _llm.invoke(prompt).content.strip().lower().split()[0]
    if label not in ("simple", "complex", "calculation", "out_of_scope"):
        label = "simple"
    return {"classification": label}


def decompose(state: State) -> dict:
    prompt = (
        "Розбий на 2-3 конкретних підпитання до ПДР України. "
        "Відповідь — JSON-список рядків, без пояснень.\n\n"
        f"Питання: {state['query']}"
    )
    try:
        sub_queries = json.loads(_llm.invoke(prompt).content)
    except Exception:
        sub_queries = [state["query"]]
    return {"sub_queries": sub_queries}


def rag_node(state: State) -> dict:
    queries = state.get("sub_queries") or [state["query"]]
    answers = [
        rag_graph.invoke({"query": q, "docs": [], "answer": ""})["answer"]
        for q in queries
    ]
    return {"rag_answers": answers}


def calculator_node(state: State) -> dict:
    prompt = (
        "Витягни зі запиту швидкість і тип розрахунку. "
        'Відповідь JSON: {"speed": N, "action": "stopping_distance" або "safe_distance"}\n\n'
        f"Запит: {state['query']}"
    )
    try:
        params = json.loads(_llm.invoke(prompt).content)
        result = traffic_calculator(int(params["speed"]), params["action"])
    except Exception:
        result = "Не вдалося розпізнати параметри для розрахунку."
    return {"web_results": [result]}


def web_search(state: State) -> dict:
    with DDGS() as ddgs:
        results = [r["body"] for r in ddgs.text(state["query"], max_results=3)]
    return {"web_results": results or ["Результатів не знайдено."]}


def synthesize(state: State) -> dict:
    rag = "\n\n".join(state.get("rag_answers") or [])
    web = "\n\n".join(state.get("web_results") or [])
    parts = []
    if rag:
        parts.append(f"Дані з ПДР:\n{rag}")
    if web:
        parts.append(f"Додаткові дані:\n{web}")
    prompt = (
        "Ти асистент з ПДР України. Дай вичерпну відповідь українською, "
        "посилайся на конкретні пункти.\n\n"
        + "\n\n".join(parts)
        + f"\n\nПитання: {state['query']}"
    )
    return {
        "final_answer": _llm.invoke(prompt).content,
        "iterations": state.get("iterations", 0) + 1,
    }


# --- graph ---

builder = StateGraph(State)
builder.add_node("classify", classify)
builder.add_node("decompose", decompose)
builder.add_node("rag_node", rag_node)
builder.add_node("calculator_node", calculator_node)
builder.add_node("web_search", web_search)
builder.add_node("synthesize", synthesize)

builder.set_entry_point("classify")
builder.add_conditional_edges("classify", route_classify)
builder.add_edge("decompose", "rag_node")
builder.add_conditional_edges("rag_node", route_rag)
builder.add_edge("calculator_node", "synthesize")
builder.add_edge("web_search", "synthesize")
builder.add_conditional_edges("synthesize", route_synthesize)

graph = builder.compile()

if __name__ == "__main__":
    result = graph.invoke({
        "query": "Яка максимальна швидкість у населеному пункті?",
        "classification": "", "sub_queries": [], "rag_answers": [],
        "web_results": [], "final_answer": "", "iterations": 0,
    })
    print(result["final_answer"])
