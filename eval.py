from core.agent import graph

EVAL_SET = [
    # (question, expected_keywords)
    # TODO: fill 15 Q&A pairs
]

if __name__ == "__main__":
    for question, keywords in EVAL_SET:
        result = graph.invoke({"query": question, "sub_queries": [], "rag_answers": [],
                               "web_results": [], "final_answer": "", "iterations": 0})
        hit = any(kw.lower() in result["final_answer"].lower() for kw in keywords)
        print(f"{'✓' if hit else '✗'} {question[:60]}")
