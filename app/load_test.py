import time
from core.agent import graph

QUERIES = [
    "Що означає розмітка 1.5?",                              # simple
    "Правила зупинки в місті?",                              # simple
    "Яка максимальна швидкість у населеному пункті?",        # simple
    "Хто має перевагу на нерегульованому перехресті?",       # simple
    "Який гальмівний шлях при швидкості 90 км/г?",          # calculation
] * 20  # 100 total

if __name__ == "__main__":
    times = []
    for i, q in enumerate(QUERIES, 1):
        t0 = time.perf_counter()
        graph.invoke({
            "query": q, "classification": "", "sub_queries": [],
            "rag_answers": [], "web_results": [], "final_answer": "", "iterations": 0,
        })
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"[{i:3}/{len(QUERIES)}] {elapsed:.1f}s — {q[:50]}", flush=True)

    s = sorted(times)
    n = len(s)
    print(f"\nn={n}  p50={s[n//2]:.1f}s  p95={s[int(n*.95)]:.1f}s  p99={s[int(n*.99)]:.1f}s")
    print(f"avg={sum(times)/n:.1f}s  min={s[0]:.1f}s  max={s[-1]:.1f}s")
